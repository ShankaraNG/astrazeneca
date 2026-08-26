import os
import json
import numpy as np
import pandas as pd
import ml_build.utils as ut
from app.logger import get_logger

log = get_logger('Explanation')

LAMBDA_VAL = 1.5

IDENTITY_COLS = ['cell_line_name', 'stripped_cell_line_name', 'ModelID', 'CVCL_ID',
                 'CCLE_Name', 'primary_disease', 'lineage']
SCORE_COLS = ['final_score', 'net_similarity', 'net_evidence', 'confidence_score',
              'modality_score', 'agreement', 'target_evidence', 'target_similarity',
              'exclusion_evidence', 'exclusion_percentile', 'exclusion_penalty',
              'exclusion_similarity', 'exclusion_similarity_percentile',
              'exclusion_similarity_penalty']
FLAG_COLS = ['fusion_flag', 'mutation_flag', 'biological_sub_group']


def _methodology_text(exclusionapplied, diseasename, fusionfilter, mutationfilter,
                      targetgenelist, exclusiongenelist):
    try:
        target_str = ", ".join(targetgenelist) if targetgenelist else "(gene symbols not supplied)"
        exclusion_str = ", ".join(exclusiongenelist) if exclusiongenelist else "none"

        lines = []
        lines.append("HOW THE CELL LINES WERE SCORED AND SELECTED")
        lines.append("")
        lines.append("The user is choosing cell lines for a target-gene study. Inputs:")
        lines.append(f"  - TARGET genes: {target_str}")
        lines.append(f"  - EXCLUSION genes: {exclusion_str}")
        lines.append(f"  - Disease restriction: {diseasename if diseasename else 'none (all lineages considered)'}")
        lines.append(f"  - Fusion filter active: {bool(fusionfilter)}")
        lines.append(f"  - Mutation filter active: {bool(mutationfilter)}")
        lines.append("")
        lines.append("SCORING METHODOLOGY:")
        lines.append("- target_evidence: mean expression/abundance across DepMap RNA, HPA RNA, and Proteomics (MinMax 0..1).")
        lines.append("- target_similarity: mean cosine similarity of co-expression profile to target genes.")
        lines.append("- confidence_score = (modality_score + agreement) / 2")
        lines.append("    * modality_score: fraction of requested genes/proteins found in datasets (1.0 = complete omics, 0.6667 = missing proteomics).")
        lines.append("    * agreement: consistency across available omics modalities.")
        if exclusionapplied:
            lines.append(f"- net_evidence = target_evidence x e^(-{LAMBDA_VAL} x percentile_exclusion_evidence)")
            lines.append(f"- net_similarity = target_similarity x e^(-{LAMBDA_VAL} x percentile_exclusion_similarity)")
        else:
            lines.append("- net_evidence = target_evidence")
            lines.append("- net_similarity = target_similarity")
        lines.append("- final_score = net_similarity x net_evidence x confidence_score")
        return "\n".join(lines)
    except Exception as e:
        log.error(f"Error building methodology text: {e}")
        raise


def _format_top10_rows_for_prompt(final_top10_df):

    try:
        df = final_top10_df.copy().sort_values(by='final_score', ascending=False).reset_index(drop=True).head(10)
        
        blocks = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            name = str(row.get('cell_line_name', 'Unknown'))
            disease = str(row.get('primary_disease', 'Unknown'))
            
            f_score = round(float(row.get('final_score', 0.0)), 4)
            n_evid = round(float(row.get('net_evidence', row.get('target_evidence', 0.0))), 4)
            n_sim = round(float(row.get('net_similarity', row.get('target_similarity', 0.0))), 4)
            conf = round(float(row.get('confidence_score', 0.0)), 4)
            mod = round(float(row.get('modality_score', 0.0)), 4)
            
            data_status = "Complete multi-omics data" if mod >= 0.99 else f"Incomplete data (modality_score = {mod:.4f}, missing proteomics abundance)"
            
            block = (
                f"Data for Rank {rank} - {name} ({disease}):\n"
                f"  - Metrics: final_score: {f_score:.4f}, net_evidence: {n_evid:.4f}, net_similarity: {n_sim:.4f}, confidence_score: {conf:.4f}, modality_score: {mod:.4f}\n"
                f"  - Status: {data_status}\n"
            )
            blocks.append(block)
            
        return "\n".join(blocks)
    except Exception as e:
        log.error(f"Error formatting top 10 rows: {e}")
        raise


def _build_messages(final_top10_df, exclusionapplied, diseasename,
                    fusionfilter, mutationfilter, targetgenelist, exclusiongenelist):
    try:
        methodology = _methodology_text(exclusionapplied, diseasename, fusionfilter,
                                        mutationfilter, targetgenelist, exclusiongenelist)
        formatted_table = _format_top10_rows_for_prompt(final_top10_df)

        df = final_top10_df.copy().sort_values(by='final_score', ascending=False).reset_index(drop=True).head(10)
        target_str = ", ".join(targetgenelist) if targetgenelist else "the requested target gene"

        outline_slots = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            name = str(row.get('cell_line_name', 'Unknown'))
            disease = str(row.get('primary_disease', 'Lymphoma/Leukemia'))
            outline_slots.append(f"Rank {rank} - {name} ({disease})\n[Write 1 detailed narrative paragraph for Rank {rank} here]")
            
        ranks_scaffold = "\n\n".join(outline_slots)

        system_prompt = (
            "You are a Senior Computational Oncologist writing a detailed, rank-by-rank cell line selection report.\n\n"
            "STRICT COMPLIANCE RULES:\n"
            "1. YOU MUST WRITE AN INDIVIDUAL PARAGRAPH FOR EVERY SINGLE RANK FROM RANK 1 TO RANK 10.\n"
            "2. DO NOT SUMMARIZE OR GROUP CELL LINES TOGETHER. Fill out EVERY rank heading in the outline.\n"
            "3. NO BULLET LISTS OF METRICS: Do NOT output lists like '* final_score: 0.7735'. Embed numbers naturally within sentences inside parentheses.\n"
            "4. HUMAN NARRATIVE COMPARISON: Explain metric relationships in natural language (e.g., 'While its co-expression similarity is exceptionally high at 0.9225, its overall final_score of 0.4883 is reduced by lower direct expression evidence of 0.6354 and missing proteomics data').\n"
            "5. NO MARKDOWN HEADERS: Do not use '#' or '##'. Use simple line headers."
        )

        user_prompt = (
            f"{methodology}\n\n"
            f"METRICS FOR THE TOP 10 CELL LINES:\n"
            f"{formatted_table}\n\n"
            f"TASK INSTRUCTIONS:\n"
            f"Write a comprehensive report evaluating cell line selection for targeting {target_str}. "
            f"You MUST strictly follow the outline template below, replacing each bracketed instruction with a fluent narrative paragraph.\n\n"
            f"STRICT OUTLINE TO FILL OUT:\n\n"
            f"Overview\n"
            f"[Write 1 paragraph summarizing the selection rationale for these top 10 cell lines.]\n\n"
            f"{ranks_scaffold}\n\n"
            f"Actionable Caveats for Laboratory Validation\n"
            f"1. [Actionable caveat 1]\n"
            f"2. [Actionable caveat 2]\n"
            f"3. [Actionable caveat 3]\n"
        )
        return system_prompt, user_prompt
    except Exception as e:
        log.error(f"Error building LLM messages: {e}")
        raise


def ollama_chat(system_prompt, user_prompt, model='llama3.2:3b',
                host='http://localhost:11434'):
    try:
        import requests
        log.info(f"ollama_chat: calling model '{model}' at {host}")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,      
                "top_p": 0.9,
                "num_predict": 4000       
            },
        }
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=1200)
        if resp.status_code != 200:
            log.error(f"ollama_chat: Ollama {resp.status_code} body: {resp.text}")
            resp.raise_for_status()
        data = resp.json()
        content = data.get('message', {}).get('content', '')
        if not content:
            raise ValueError("Ollama returned an empty completion.")
        log.info(f"ollama_chat: received {len(content)} character(s)")
        return content
    except Exception as e:
        log.error(f"Error calling Ollama chat backend: {e}")
        raise


def generate_explanation(final_top10_df, targetgenelist=None, exclusiongenelist=None,
                         diseasename=None, fusionfilter=False, mutationfilter=False,
                         llm_callable=None, model='llama3.2:3b',
                         host='http://localhost:11434', save=False):
    try:
        log.info("generate_explanation: starting")
        if final_top10_df is None or final_top10_df.empty:
            raise ValueError("final_top10_df is empty or undefined.")

        exclusionapplied = bool(exclusiongenelist) or ('exclusion_penalty' in final_top10_df.columns)

        system_prompt, user_prompt = _build_messages(
            final_top10_df, exclusionapplied, diseasename,
            fusionfilter, mutationfilter, targetgenelist, exclusiongenelist)

        caller = llm_callable if llm_callable is not None else ollama_chat
        if llm_callable is not None:
            explanation = caller(system_prompt, user_prompt)
        else:
            explanation = caller(system_prompt, user_prompt, model=model, host=host)

        if save:
            out_df = pd.DataFrame([{"explanation": explanation}])
            result = ut.data_save(out_df, 'test', 'top10', 'top10_explanation.csv')
            if result != "successfull":
                log.warning("generate_explanation: save did not report success.")
        return explanation
    except Exception as e:
        log.error(f"Error generating top-10 explanation: {e}")
        raise