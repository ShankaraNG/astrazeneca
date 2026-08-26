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
              'modality_score', 'agreement', 'target_similarity', 'target_evidence',
              'exclusion_evidence', 'exclusion_percentile', 'exclusion_penalty',
              'exclusion_similarity', 'exclusion_similarity_percentile',
              'exclusion_similarity_penalty']
FLAG_COLS = ['fusion_flag', 'mutation_flag', 'biological_sub_group']


def _methodology_text(exclusionapplied, diseasename, fusionfilter, mutationfilter,
                      targetgenelist, exclusiongenelist):
    try:
        target_str = ", ".join(targetgenelist) if targetgenelist else "(gene symbols not supplied to this function)"
        exclusion_str = ", ".join(exclusiongenelist) if exclusiongenelist else "none"

        lines = []
        lines.append("HOW THE CELL LINES WERE SCORED AND SELECTED")
        lines.append("")
        lines.append("The user is choosing cell lines for a target-gene study. They supplied:")
        lines.append(f"  - TARGET genes (want expressed / relevant): {target_str}")
        lines.append(f"  - EXCLUSION genes (want absent / avoided): {exclusion_str}")
        lines.append(f"  - Disease restriction: {diseasename if diseasename else 'none (all lineages considered)'}")
        lines.append(f"  - Fusion filter active: {bool(fusionfilter)}")
        lines.append(f"  - Mutation filter active: {bool(mutationfilter)}")
        lines.append("")
        lines.append("PIPELINE (offline models were trained beforehand; this is the query-time flow):")
        lines.append("1. Each target gene symbol is resolved to an Ensembl gene id (ENSG), and where "
                     "possible to a protein id.")
        lines.append("2. For every cell line we read four omics views: DepMap RNA expression, HPA RNA "
                     "expression, harmonized proteomics abundance, and a precomputed gene-gene cosine "
                     "similarity matrix. All values were MinMax-scaled during model build, so every "
                     "score below sits on a 0..1 scale.")
        lines.append("")
        lines.append("SCORES (all per cell line):")
        lines.append("- target_evidence: mean of the target genes' expression/abundance values across "
                     "the omics views. HIGH = the cell line strongly expresses the genes of interest. "
                     "A gene id that was not found in a table contributes 0 to this mean.")
        lines.append("- target_similarity: mean cosine similarity between the cell line and the target "
                     "genes' co-expression signature. HIGH = the cell line's overall profile is "
                     "biologically aligned with the targets, beyond raw expression.")
        lines.append("- confidence_score = (modality_score + agreement) / 2:")
        lines.append("    * modality_score = fraction of the requested gene/protein ids that were "
                     "actually found in the data (data completeness). LOW means much of the requested "
                     "evidence was missing for that cell line.")
        lines.append("    * agreement = 1 / (1 + variance) across the found values (consistency). HIGH "
                     "means the different omics views agree with each other for that cell line.")
        lines.append("    Confidence pools BOTH target and exclusion gene/protein ids.")
        if exclusionapplied:
            lines.append("- EXCLUSION penalties (because exclusion genes were supplied):")
            lines.append(f"    * exclusion_penalty = e^(-{LAMBDA_VAL} x percentile_of_exclusion_evidence). "
                         "A cell line that strongly expresses the exclusion genes sits at a high "
                         "percentile and is pushed toward 0 (heavily penalised).")
            lines.append(f"    * exclusion_similarity_penalty = e^(-{LAMBDA_VAL} x "
                         "percentile_of_exclusion_similarity), same idea on the similarity side.")
            lines.append("- net_evidence   = target_evidence   x exclusion_penalty")
            lines.append("- net_similarity = target_similarity x exclusion_similarity_penalty")
            lines.append("  (Exclusion therefore acts on BOTH evidence and similarity, by design.)")
        else:
            lines.append("- No exclusion genes were supplied, so:")
            lines.append("- net_evidence   = target_evidence")
            lines.append("- net_similarity = target_similarity")
        lines.append("")
        lines.append("- final_score = net_similarity x net_evidence x confidence_score")
        lines.append("  A cell line ranks highly only if it is BOTH strongly relevant (evidence + "
                     "similarity) AND well-supported by consistent, complete data (confidence). A low "
                     "value in any one factor drags the product down.")
        lines.append("")
        lines.append("FLAGS AND FILTERS:")
        lines.append("- fusion_flag = 'Yes' if the cell line has a fusion event where BOTH partner "
                     "genes are in the supplied gene list.")
        lines.append("- mutation_flag = 'Yes' if the cell line carries a somatic mutation in any gene "
                     "in the supplied gene list.")
        if fusionfilter:
            lines.append("- The FUSION FILTER was ON: fusion-positive cell lines were REMOVED before "
                         "ranking, so every selected line is fusion-negative for these genes.")
        if mutationfilter:
            lines.append("- The MUTATION FILTER was ON: mutation-positive cell lines were REMOVED "
                         "before ranking, so every selected line is mutation-negative for these genes.")
        if not fusionfilter and not mutationfilter:
            lines.append("- Neither filter was applied as a removal step; the flags are informational "
                         "only and flagged lines were still eligible for selection.")
        lines.append("")
        lines.append("- biological_sub_group: the KMeans cluster (on MOFA multi-omics factors) the "
                     "cell line belongs to, i.e. its multi-omics neighbourhood.")
        lines.append("- The top 10 were taken by descending final_score after all flagging/filtering.")
        return "\n".join(lines)
    except Exception as e:
        log.error(f"Error building methodology text: {e}")
        raise


def _row_digest(final_top10_df):
    try:
        df = final_top10_df.copy().reset_index(drop=True)
        id_present = [c for c in IDENTITY_COLS if c in df.columns]
        score_present = [c for c in SCORE_COLS if c in df.columns]
        flag_present = [c for c in FLAG_COLS if c in df.columns]
        keep = list(dict.fromkeys(id_present + score_present + flag_present))
        slim = df[keep] if keep else df

        records = []
        for rank, (_, row) in enumerate(slim.iterrows(), start=1):
            entry = {"rank": rank}
            for c in keep:
                val = row[c]
                if isinstance(val, (int, float, np.floating)):
                    fval = float(val)
                    entry[c] = round(fval, 4) if np.isfinite(fval) else None
                elif pd.isna(val):
                    entry[c] = None
                else:
                    entry[c] = str(val)
            records.append(entry)
        log.info(f"_row_digest: built {len(records)} record(s) with columns {keep}")
        return records
    except Exception as e:
        log.error(f"Error building row digest: {e}")
        raise


def _population_context(final_mutated_df):
    try:
        ctx = {}
        ctx['total_candidates'] = int(len(final_mutated_df))
        if 'final_score' in final_mutated_df.columns:
            fs = final_mutated_df['final_score'].dropna()
            fs = fs[np.isfinite(fs)]
            if not fs.empty:
                ctx['final_score_min'] = round(float(fs.min()), 4)
                ctx['final_score_max'] = round(float(fs.max()), 4)
                ctx['final_score_median'] = round(float(fs.median()), 4)
        for flag in ['fusion_flag', 'mutation_flag']:
            if flag in final_mutated_df.columns:
                ctx[f'{flag}_yes_count'] = int((final_mutated_df[flag] == 'Yes').sum())
        log.info(f"_population_context: {ctx}")
        return ctx
    except Exception as e:
        log.error(f"Error building population context: {e}")
        raise


def _build_messages(final_top10_df, final_mutated_df, exclusionapplied, diseasename,
                    fusionfilter, mutationfilter, targetgenelist, exclusiongenelist):
    try:
        methodology = _methodology_text(exclusionapplied, diseasename, fusionfilter,
                                        mutationfilter, targetgenelist, exclusiongenelist)
        digest = _row_digest(final_top10_df)
        population = _population_context(final_mutated_df)

        target_str = ", ".join(targetgenelist) if targetgenelist else "the supplied target genes"

        system_prompt = (
            "You are a computational biology assistant explaining the output of a multi-omics "
            "cell line recommendation system to a cancer researcher. You have two jobs: (1) explain "
            "the SYSTEM's scoring — grounded strictly in the scores and methodology given, never "
            "claiming a score means something the methodology does not say; and (2) add BIOLOGICAL "
            "CONTEXT for the cell lines and genes involved, drawing on well-established biology. "
            "Use each cell line's real 'cell_line_name', 'primary_disease' and 'lineage' from the "
            "data as the source of truth for what it is. The 'ModelID' (ACH-xxxxxx) is only a "
            "database accession, NOT a biological identity — never infer tissue, disease or biology "
            "from the ModelID. If you are not confident about specifics for a named line, stay with "
            "what its primary_disease and lineage establish and say the rest is uncertain. Never "
            "fabricate specific mutations, drug responses, or identifiers, and never assign the same "
            "tissue to every line — each row has its own primary_disease."
        )

        user_prompt = (
            f"{methodology}\n\n"
            f"POPULATION CONTEXT (the full scored candidate set):\n"
            f"{json.dumps(population, indent=2)}\n\n"
            f"THE SELECTED TOP 10 (already ranked by final_score):\n"
            f"{json.dumps(digest, indent=2)}\n\n"
            "TASK:\n"
            "1. Open with a short paragraph on the basis these 10 were chosen as a group, "
            "referencing the scoring logic (evidence, similarity, confidence, and any exclusion "
            "penalties / filters that were active).\n"
            "2. Then, for EACH of the 10 cell lines in rank order, write a detailed paragraph "
            "covering two things in flowing prose:\n"
            "   - why it ranked where it did: cite its actual scores, what drove them up (high "
            "evidence/similarity, strong confidence) or held them back (low modality = missing "
            "data, low agreement = inconsistent omics, exclusion penalty), and note its "
            "fusion/mutation flags and biological sub-group cluster.\n"
            f"   - its biology: use the line's real cell_line_name, primary_disease and lineage "
            f"from the data to describe its tissue and cancer type, what it is commonly used to "
            f"model, and how the target genes ({target_str}) relate to that biology. Give several "
            "sentences of genuine detail when you reliably know the named line; when unsure, keep "
            "to what primary_disease/lineage establish and flag the rest as uncertain. Do NOT "
            "repeat the same tissue for different lines — each has its own primary_disease.\n"
            "3. Close with two or three honest caveats: what these scores can and cannot tell the "
            "researcher, and a reminder that the biological notes are background knowledge to "
            "verify, not outputs of this system.\n\n"
            "OUTPUT FORMAT (follow exactly):\n"
            "- Write in plain prose paragraphs. Do NOT use Markdown headings (no '#', '##', '###').\n"
            "- Do NOT wrap anything in asterisks (no '**bold**', no '*italics*'). Labels must be "
            "plain text.\n"
            "- Begin each cell line's paragraph with its rank and real NAME and disease, for "
            "example: 'Rank 1 - AU565 (breast cancer).' Never lead with the ACH ModelID.\n"
            "- Introduce biological background naturally with a phrase like 'In general,' inside "
            "the same paragraph, rather than a separate bold label or header.\n"
            "- Write for a knowledgeable cancer-biology researcher: specific, numeric where it "
            "helps, and flowing rather than bulleted."
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
            "options": {"temperature": 0.3},
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


def generate_explanation(final_top10_df, final_mutated_df, targetgenelist=None,
                         exclusiongenelist=None, diseasename=None, fusionfilter=False,
                         mutationfilter=False, llm_callable=None, model='llama3.2:3b',
                         host='http://localhost:11434', save=True):
    try:
        log.info("generate_explanation: starting")
        if final_top10_df is None or final_top10_df.empty:
            raise ValueError("final_top10_df is empty or undefined.")
        if final_mutated_df is None or final_mutated_df.empty:
            raise ValueError("final_mutated_df is empty or undefined.")

        # Infer whether exclusion was in play if the caller didn't tell us.
        exclusionapplied = bool(exclusiongenelist) or ('exclusion_penalty' in final_top10_df.columns)
        log.info(f"generate_explanation: exclusionapplied={exclusionapplied} "
                 f"fusionfilter={bool(fusionfilter)} mutationfilter={bool(mutationfilter)} "
                 f"disease={diseasename}")

        system_prompt, user_prompt = _build_messages(
            final_top10_df, final_mutated_df, exclusionapplied, diseasename,
            fusionfilter, mutationfilter, targetgenelist, exclusiongenelist)
        log.info(f"generate_explanation: prompt built "
                 f"(system={len(system_prompt)} chars, user={len(user_prompt)} chars)")

        caller = llm_callable if llm_callable is not None else ollama_chat
        if llm_callable is not None:
            explanation = caller(system_prompt, user_prompt)
        else:
            explanation = caller(system_prompt, user_prompt, model=model, host=host)

        log.info(f"generate_explanation: explanation generated ({len(explanation)} chars)")

        if save:
            out_df = pd.DataFrame([{"explanation": explanation}])
            result = ut.data_save(out_df, 'test', 'top10', 'top10_explanation.csv')
            if result != "successfull":
                log.warning("generate_explanation: explanation save did not report success.")
        return explanation
    except Exception as e:
        log.error(f"Error generating the top-10 explanation: {e}")
        raise