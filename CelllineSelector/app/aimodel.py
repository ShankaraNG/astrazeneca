import re
import pandas as pd

import ml_build.utils as ut
from app.logger import get_logger

log = get_logger('Explanation')

LAMBDA_VAL = 1.5
SEED = 42
MAX_ATTEMPTS = 2



def _methodology_text(exclusionapplied, diseasename, fusionfilter,
                      mutationfilter, targetgenelist, exclusiongenelist):
    try:
        target_str = ", ".join(targetgenelist) if targetgenelist \
            else "(gene symbols not supplied)"
        exclusion_str = ", ".join(exclusiongenelist) if exclusiongenelist \
            else "none"
        disease_str = diseasename if diseasename \
            else "none (all lineages considered)"

        lines = [
            "HOW THE CELL LINES WERE SCORED AND SELECTED",
            "",
            "The user is choosing cell lines for a target-gene study. Inputs:",
            f"  - TARGET genes: {target_str}",
            f"  - EXCLUSION genes: {exclusion_str}",
            f"  - Disease restriction: {disease_str}",
            f"  - Fusion filter active: {bool(fusionfilter)}",
            f"  - Mutation filter active: {bool(mutationfilter)}",
            "",
            "SCORING METHODOLOGY:",
            "- target_evidence: mean expression/abundance across DepMap RNA, "
            "HPA RNA, and Proteomics (MinMax 0..1).",
            "- target_similarity: mean cosine similarity of co-expression "
            "profile to target genes.",
            "- confidence_score = (modality_score + agreement) / 2",
            "    * modality_score: fraction of requested genes/proteins found "
            "in datasets (1.0 = complete omics, 0.6667 = missing proteomics).",
            "    * agreement: consistency across available omics modalities.",
        ]

        if exclusionapplied:
            lines.append(
                f"- net_evidence = target_evidence x "
                f"e^(-{LAMBDA_VAL} x percentile_exclusion_evidence)")
            lines.append(
                f"- net_similarity = target_similarity x "
                f"e^(-{LAMBDA_VAL} x percentile_exclusion_similarity)")
        else:
            lines.append("- net_evidence = target_evidence "
                         "(no exclusion genes supplied, so no penalty applied)")
            lines.append("- net_similarity = target_similarity "
                         "(no exclusion genes supplied, so no penalty applied)")

        lines.append("- final_score = net_similarity x net_evidence x "
                     "confidence_score")
        return "\n".join(lines)
    except Exception as e:
        log.error(f"Error building methodology text: {e}")
        raise


def _num(row, key, default=0.0):
    val = row.get(key, None)
    if val is None or pd.isna(val):
        return float(default)
    return float(val)


def _text(row, key, default='Not specified'):
    val = row.get(key, None)
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return default
    s = str(val).strip()
    return s if s else default


def _score_driver(n_evid, n_sim, conf):
    parts = {"evidence": n_evid, "similarity": n_sim, "confidence": conf}
    strongest = max(parts, key=parts.get)
    weakest = min(parts, key=parts.get)

    if parts[strongest] - parts[weakest] < 0.05:
        return (f"all three components are close in value "
                f"({parts[weakest]:.4f}-{parts[strongest]:.4f}), so none "
                f"dominates the final score")

    return (f"the {strongest} component is the highest at "
            f"{parts[strongest]:.4f}; the {weakest} component is the "
            f"lowest at {parts[weakest]:.4f} and, because the final score "
            f"is the product of the three, it is the factor limiting this "
            f"cell line's score")


def _relative_label(value, series):
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-9:
        return "the same as every other line shown"
    pos = (float(value) - lo) / (hi - lo)
    if pos >= 0.75:
        return "among the highest of the ten shown"
    if pos >= 0.4:
        return "mid-range among the ten shown"
    return "among the lowest of the ten shown"


def _confidence_reason(mod, agree):
    if mod < 0.99 and agree < 0.8:
        return (f"both inputs are reduced: proteomics abundance is missing "
                f"(modality {mod:.4f}) and the available modalities "
                f"disagree (agreement {agree:.4f})")
    if mod < 0.99:
        return (f"proteomics abundance is missing for this line, reducing "
                f"the modality score to {mod:.4f}")
    if agree < 0.8:
        return (f"the available omics modalities disagree with each other "
                f"(agreement {agree:.4f})")
    return (f"both inputs are close to complete (modality {mod:.4f}, "
            f"agreement {agree:.4f})")


def _comparison_to_previous(row, prev):
    if prev is None:
        return None

    comps = {
        "evidence": (_num(row, 'net_evidence', _num(row, 'target_evidence')),
                     _num(prev, 'net_evidence', _num(prev, 'target_evidence'))),
        "similarity": (_num(row, 'net_similarity', _num(row, 'target_similarity')),
                       _num(prev, 'net_similarity', _num(prev, 'target_similarity'))),
        "confidence": (_num(row, 'confidence_score'),
                       _num(prev, 'confidence_score')),
    }
    deltas = {k: v[0] - v[1] for k, v in comps.items()}
    worst = min(deltas, key=deltas.get)

    prev_name = _text(prev, 'cell_line_name', 'the line above')
    if deltas[worst] >= -1e-9:
        return (f"every component is at least as high as {prev_name}'s, so "
                f"the small difference in final score comes from their "
                f"combination rather than any single component")
    return (f"compared with {prev_name}, the {worst} component is lower "
            f"({comps[worst][0]:.4f} against {comps[worst][1]:.4f}), which "
            f"is the largest single contributor to the lower final score "
            f"and hence the lower rank")


def _format_rows_for_prompt(df):
    try:
        ev_series = df.apply(
            lambda r: _num(r, 'net_evidence', _num(r, 'target_evidence')), axis=1)
        sim_series = df.apply(
            lambda r: _num(r, 'net_similarity', _num(r, 'target_similarity')), axis=1)
        conf_series = df['confidence_score'] if 'confidence_score' in df.columns \
            else df.apply(lambda r: _num(r, 'confidence_score'), axis=1)

        blocks = []
        prev = None
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            name = _text(row, 'cell_line_name', 'Unknown')
            disease = _text(row, 'primary_disease', 'Unknown')
            lineage = _text(row, 'lineage')
            subgroup = _text(row, 'biological_sub_group')

            f_score = _num(row, 'final_score')
            n_evid = _num(row, 'net_evidence', _num(row, 'target_evidence'))
            n_sim = _num(row, 'net_similarity', _num(row, 'target_similarity'))
            conf = _num(row, 'confidence_score')
            mod = _num(row, 'modality_score')
            agree = _num(row, 'agreement')

            data_status = ("Complete multi-omics data" if mod >= 0.99 else
                           f"Incomplete data (modality_score = {mod:.4f}, "
                           "missing proteomics abundance)")

            mut = _text(row, 'mutation_flag', 'none recorded')
            fus = _text(row, 'fusion_flag', 'none recorded')

            block = (
                f"Data for Rank {rank} - {name} ({disease}):\n"
                f"  - Evidence score: {n_evid:.4f} "
                f"({_relative_label(n_evid, ev_series)})\n"
                f"  - Similarity score: {n_sim:.4f} "
                f"({_relative_label(n_sim, sim_series)})\n"
                f"  - Confidence score: {conf:.4f} "
                f"({_relative_label(conf, conf_series)})\n"
                f"  - Final score: {f_score:.4f} "
                f"(the product of the three components above)\n"
                f"  - Why the confidence score is what it is: "
                f"{_confidence_reason(mod, agree)}\n"
                f"  - Limiting component: "
                f"{_score_driver(n_evid, n_sim, conf)}\n")

            comp = _comparison_to_previous(row, prev)
            if comp:
                block += f"  - Comparison with the rank above: {comp}\n"

            block += (
                f"  - Data completeness: {data_status}\n"
                f"  - Biological context (USE ONLY THIS): "
                f"primary disease = {disease}; lineage = {lineage}; "
                f"biological subgroup = {subgroup}\n"
                f"  - Mutation flag: {mut}\n"
                f"  - Fusion flag: {fus}\n")

            blocks.append(block)
            prev = row

        return "\n".join(blocks)
    except Exception as e:
        log.error(f"Error formatting rows for prompt: {e}")
        raise


def _rank_of(value, series):
    return int((series > float(value)).sum()) + 1


def _place(r, n):
    if r == 1:
        return "the highest of the ten"
    if r == n:
        return "the lowest of the ten"
    return f"{r}{'st' if r == 1 else 'nd' if r == 2 else 'rd' if r == 3 else 'th'} highest of the ten"


COMPONENT_MEANING = {
    "evidence": ("it reflects mean expression and abundance of the target "
                 "across DepMap RNA, HPA RNA and proteomics"),
    "similarity": ("it reflects how closely this line's co-expression "
                   "profile matches the target gene"),
}


def render_report(df, targetgenelist=None, exclusiongenelist=None,
                  diseasename=None, fusionfilter=False, mutationfilter=False):
    n = len(df)
    ev = df.apply(lambda r: _num(r, 'net_evidence',
                                 _num(r, 'target_evidence')), axis=1)
    sim = df.apply(lambda r: _num(r, 'net_similarity',
                                  _num(r, 'target_similarity')), axis=1)
    conf = df.apply(lambda r: _num(r, 'confidence_score'), axis=1)

    target_str = ", ".join(targetgenelist) if targetgenelist \
        else "the requested target gene"

    out = ["Overview",
           f"The {n} cell lines below are ranked by final score for "
           f"targeting {target_str}. The final score is the product of "
           f"evidence, similarity and confidence, with no weighting "
           f"applied. Components are compared against the other cell lines "
           f"shown, not the full dataset.", ""]

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        name = _text(row, 'cell_line_name', 'Unknown')
        disease = _text(row, 'primary_disease', 'Unknown')
        subgroup = _text(row, 'biological_sub_group')

        e = _num(row, 'net_evidence', _num(row, 'target_evidence'))
        s = _num(row, 'net_similarity', _num(row, 'target_similarity'))
        c = _num(row, 'confidence_score')
        f = _num(row, 'final_score')
        mod = _num(row, 'modality_score')
        agree = _num(row, 'agreement')

        ranks = {"evidence": _rank_of(e, ev),
                 "similarity": _rank_of(s, sim),
                 "confidence": _rank_of(c, conf)}
        vals = {"evidence": e, "similarity": s, "confidence": c}

        best = min(ranks, key=ranks.get)
        worst = max(ranks, key=ranks.get)

        def why(comp):
            if comp == "confidence":
                return _confidence_reason(mod, agree)
            return COMPONENT_MEANING[comp]

        sent = [
            f"Evidence {e:.4f}, similarity {s:.4f}, confidence {c:.4f}, "
            f"giving a final score of {f:.4f}."]

        if best == worst:
            sent.append(
                f"All three components sit at the same position relative "
                f"to the other lines shown.")
        else:
            sent.append(
                f"Its strongest component is {best} at {vals[best]:.4f}, "
                f"{_place(ranks[best], n)}; {why(best)}.")
            sent.append(
                f"Its weakest is {worst} at {vals[worst]:.4f}, "
                f"{_place(ranks[worst], n)}; {why(worst)}.")

        limiting = min(vals, key=vals.get)
        sent.append(
            f"Of its own three components, {limiting} is the smallest at "
            f"{vals[limiting]:.4f}, and since the final score is their "
            f"product this is what holds the line back.")

        sent.append(f"Biological subgroup: {subgroup}.")

        out.append(f"Rank {rank} - {name} ({disease})")
        out.append(" ".join(sent))
        out.append("")

    first, last = df.iloc[0], df.iloc[-1]
    f_name = _text(first, 'cell_line_name', 'the top line')
    l_name = _text(last, 'cell_line_name', 'the last line')

    gaps = {
        "evidence": (_num(first, 'net_evidence', _num(first, 'target_evidence')),
                     _num(last, 'net_evidence', _num(last, 'target_evidence'))),
        "similarity": (_num(first, 'net_similarity', _num(first, 'target_similarity')),
                       _num(last, 'net_similarity', _num(last, 'target_similarity'))),
        "confidence": (_num(first, 'confidence_score'),
                       _num(last, 'confidence_score')),
    }
    widest = max(gaps, key=lambda k: gaps[k][0] - gaps[k][1])

    def _flagged(r, col):
        return _text(r, col, 'no').strip().lower() not in (
            'no', 'false', '0', 'none', 'none recorded', 'not specified', '')

    n_mut = sum(1 for _, r in df.iterrows() if _flagged(r, 'mutation_flag'))
    n_fus = sum(1 for _, r in df.iterrows() if _flagged(r, 'fusion_flag'))
    n_incomplete = sum(1 for _, r in df.iterrows()
                       if _num(r, 'modality_score') < 0.99)

    subs = [_text(r, 'biological_sub_group') for _, r in df.iterrows()]
    subs = [x for x in subs if x and x != 'Not specified']

    out.append("Actionable Caveats for Laboratory Validation")
    out.append(
        f"1. {f_name} ranks first ({_num(first, 'final_score'):.4f}) and "
        f"{l_name} ranks last ({_num(last, 'final_score'):.4f}). They "
        f"differ most in {widest} ({gaps[widest][0]:.4f} against "
        f"{gaps[widest][1]:.4f}).")

    filt = [x for x, on in (("mutation", mutationfilter),
                            ("fusion", fusionfilter)) if on]
    filter_note = (
        f" A {' and '.join(filt)} filter was applied to this query, so this "
        f"count may reflect the filter rather than the underlying data."
        if filt else "")
    out.append(
        f"2. Mutation and fusion flags: {n_mut} of {n} carry a mutation "
        f"flag, {n_fus} of {n} a gene fusion involving "
        f"{target_str}.{filter_note}")

    if subs:
        top_sub = max(set(subs), key=subs.count)
        out.append(
            f"3. Biological subgroup: {subs.count(top_sub)} of {n} lines "
            f"belong to subgroup {top_sub}. {n_incomplete} of {n} have "
            f"incomplete proteomics data.")
    else:
        out.append(f"3. Subgroup not recorded. {n_incomplete} of {n} have "
                   f"incomplete proteomics data.")

    return "\n".join(out)


SYSTEM_PROMPT = (
    "You are a computational oncologist writing a rank-by-rank cell line "
    "selection report.\n\n"
    "STRICT COMPLIANCE RULES:\n"
    "1. Write one individual paragraph for EVERY rank from 1 to the last "
    "rank in the outline. Do not summarise or group cell lines together.\n"
    "2. GROUNDING: Use ONLY the information supplied in the data blocks. "
    "Do NOT introduce mutation status, drug responses, growth "
    "characteristics, publication history, or any other biological fact "
    "that is not written in the block for that cell line. If the data does "
    "not say something, do not say it either. Inventing plausible-sounding "
    "detail is the single worst failure mode for this report. This applies "
    "equally to the Overview and the Caveats: do not describe what the "
    "target gene does, and do not state what a flag implies beyond the "
    "supplied definition.\n"
    "3. REPORT ALL FOUR SCORES: every paragraph must state the cell line's "
    "evidence score, similarity score, confidence score and final score, "
    "each with its exact value from the data block. Do not omit any of "
    "them and do not round differently from the block.\n"
    "4. USE THE SUPPLIED RELATIVE LABELS: each score comes with a label "
    "such as 'among the highest of the ten shown'. Use those labels and "
    "no others. Do not invent your own comparison, and do not compare a "
    "cell line's evidence score against its own confidence score - the "
    "three metrics occupy different ranges and such a comparison is "
    "meaningless.\n"
    "5. USE THE SUPPLIED EXPLANATIONS: each block contains a computed "
    "reason for the confidence score, a limiting component, and (from "
    "rank 2 onwards) a comparison with the rank above. State these. Do "
    "not derive your own explanation for why a line ranks where it does.\n"
    "6. NO INVENTED METHODOLOGY: the final score is the plain product of "
    "the three components. There are no weights, no component is "
    "'most influential' by design, and no modality is weighted above "
    "another. Do not say otherwise. Do not attribute a score to 'strong "
    "RNA expression' or any other cause not stated in the block.\n"
    "7. DO NOT ECHO THE PROMPT: never reproduce the scoring methodology, "
    "the query inputs, or the data blocks in your output. Begin your "
    "response directly with the word 'Overview' and write only the "
    "report.\n"
    "8. NO BULLET LISTS OF METRICS: do not output lines like "
    "'* final_score: 0.7735'. Embed the numbers inside sentences.\n"
    "9. Close each paragraph with 2-3 sentences on the cell line's "
    "biological context, drawn strictly from the disease, lineage, "
    "subgroup and flag fields provided for it.\n"
    "10. FLAGS: state the mutation and fusion flag values exactly as given "
    "for that cell line, and note that flags are informational and do not "
    "affect the score. In the caveats, use only the verified counts "
    "supplied to you.\n"
    "11. DATA COMPLETENESS: report the completeness status exactly as "
    "written in the block. Do not state that proteomics is missing for a "
    "cell line whose block says the data is complete.\n"
    "12. NO MARKDOWN HEADERS: do not use '#' or '##'. Use plain line "
    "headers exactly as given in the outline."
)


def _build_messages(df, exclusionapplied, diseasename, fusionfilter,
                    mutationfilter, targetgenelist, exclusiongenelist):
    try:
        methodology = _methodology_text(
            exclusionapplied, diseasename, fusionfilter, mutationfilter,
            targetgenelist, exclusiongenelist)
        data_blocks = _format_rows_for_prompt(df)

        target_str = ", ".join(targetgenelist) if targetgenelist \
            else "the requested target gene"


        n = len(df)
        n_mut = sum(1 for _, r in df.iterrows()
                    if _text(r, 'mutation_flag', 'no').strip().lower()
                    not in ('no', 'false', '0', 'none', 'none recorded',
                            'not specified', ''))
        n_fus = sum(1 for _, r in df.iterrows()
                    if _text(r, 'fusion_flag', 'no').strip().lower()
                    not in ('no', 'false', '0', 'none', 'none recorded',
                            'not specified', ''))
        n_incomplete = sum(1 for _, r in df.iterrows()
                           if _num(r, 'modality_score') < 0.99)


        first, last = df.iloc[0], df.iloc[-1]
        f_name = _text(first, 'cell_line_name', 'the top line')
        l_name = _text(last, 'cell_line_name', 'the last line')
        gaps = {
            "evidence": (_num(first, 'net_evidence', _num(first, 'target_evidence')),
                         _num(last, 'net_evidence', _num(last, 'target_evidence'))),
            "similarity": (_num(first, 'net_similarity', _num(first, 'target_similarity')),
                           _num(last, 'net_similarity', _num(last, 'target_similarity'))),
            "confidence": (_num(first, 'confidence_score'),
                           _num(last, 'confidence_score')),
        }
        widest = max(gaps, key=lambda k: gaps[k][0] - gaps[k][1])
        spread = (
            f"{f_name} (final score {_num(first, 'final_score'):.4f}) and "
            f"{l_name} (final score {_num(last, 'final_score'):.4f}) differ "
            f"most in the {widest} component "
            f"({gaps[widest][0]:.4f} against {gaps[widest][1]:.4f})")

       
        subs = [_text(r, 'biological_sub_group') for _, r in df.iterrows()]
        subs = [s for s in subs if s and s != 'Not specified']
        if subs:
            top_sub = max(set(subs), key=subs.count)
            sub_line = (f"{subs.count(top_sub)} of {n} cell lines belong to "
                        f"biological subgroup {top_sub}")
        else:
            sub_line = "biological subgroup is not recorded for these lines"

        filter_note = (
            "the mutation filter was active for this query"
            if fusionfilter or mutationfilter
            else "no mutation or fusion filter was applied to this query")

        tally = (
            f"VERIFIED FACTS FOR THE CAVEATS (use these figures exactly; do "
            f"not recount or reinterpret them):\n"
            f"  - Spread across the ranking: {spread}.\n"
            f"  - {n_mut} of {n} cell lines carry a mutation flag; "
            f"{n_fus} of {n} carry a fusion flag. Note that {filter_note}, "
            f"so this count may reflect the filter setting rather than the "
            f"underlying data.\n"
            f"  - Subgroup composition: {sub_line}.\n"
            f"  - {n_incomplete} of {n} cell lines have incomplete "
            f"proteomics data.\n"
            f"\n"
            f"FLAG MEANING (state it this way if you mention flags; do not "
            f"substitute your own definition):\n"
            f"  - A mutation flag means the cell line carries a recorded "
            f"variant in a queried gene. A fusion flag means a recorded "
            f"gene fusion involving a queried gene. Flags are "
            f"informational and do not affect any score. They do not "
            f"indicate contamination, genetic modification, or line "
            f"quality.\n")

        headers = "\n\n".join(
            f"Rank {rank} - {_text(row, 'cell_line_name', 'Unknown')} "
            f"({_text(row, 'primary_disease', 'Unknown')})"
            for rank, (_, row) in enumerate(df.iterrows(), start=1))

        user_prompt = (
            f"{methodology}\n\n"
            f"DATA FOR THE TOP {len(df)} CELL LINES:\n"
            f"{data_blocks}\n\n"
            f"{tally}\n"
            f"TASK:\n"
            f"Write a report evaluating cell line selection for targeting "
            f"{target_str}. Follow the outline below exactly, writing one "
            f"fluent narrative paragraph under each rank heading. Reproduce "
            f"the headings verbatim.\n\n"
            f"PARAGRAPH TEMPLATE - follow this order for every rank, using "
            f"only that rank's supplied facts:\n"
            f"  (a) state the evidence, similarity and confidence scores "
            f"and the final score they multiply to;\n"
            f"  (b) state how each stands relative to the other nine, using "
            f"the supplied relative labels verbatim;\n"
            f"  (c) give the supplied reason for the confidence score;\n"
            f"  (d) for ranks 2 onwards, give the supplied comparison with "
            f"the rank above;\n"
            f"  (e) close with the biological subgroup, disease and lineage, "
            f"and the flag values.\n\n"
            f"OUTLINE:\n\n"
            f"Overview\n"
            f"[One paragraph. State only that the ranking is by final "
            f"score, which is the product of evidence, similarity and "
            f"confidence. Do not describe the target gene's biology and do "
            f"not claim the components are weighted.]\n\n"
            f"{headers}\n\n"
            f"Actionable Caveats for Laboratory Validation\n"
            f"1. [State the supplied spread between the first and last "
            f"ranked line, and what that means for choosing between them.]\n"
            f"2. [State the supplied flag counts, the flag meaning, and the "
            f"note about the filter setting.]\n"
            f"3. [State the supplied subgroup composition and the "
            f"incomplete-data count.]\n"
        )
        return SYSTEM_PROMPT, user_prompt
    except Exception as e:
        log.error(f"Error building LLM messages: {e}")
        raise


def ollama_chat(system_prompt, user_prompt, model='llama3.2:3b',
                host='http://localhost:11434', timeout=1200, seed=SEED):
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
                "seed": seed,
                "num_predict": 4000,
            },
        }
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        if resp.status_code != 200:
            log.error(f"ollama_chat: Ollama {resp.status_code} "
                      f"body: {resp.text}")
            resp.raise_for_status()

        content = resp.json().get('message', {}).get('content', '')
        if not content:
            raise ValueError("Ollama returned an empty completion.")
        log.info(f"ollama_chat: received {len(content)} character(s)")
        return content
    except Exception as e:
        log.error(f"Error calling Ollama chat backend: {e}")
        raise



def _strip_echoed_prompt(text):
    m = re.search(r'(?im)^\s*Overview\s*$', text)
    if m and m.start() > 0:
        return text[m.start():].lstrip()
    return text


def _normalise(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _missing_ranks(explanation, df):
    haystack = _normalise(explanation)
    missing = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        name = _normalise(_text(row, 'cell_line_name', ''))
        if not name or name not in haystack:
            missing.append(rank)
    return missing



REPHRASE_SYSTEM_PROMPT = (
    "You are an editor. You will be given a factually verified cell line "
    "selection report. Rewrite it as fluent, readable prose.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Do NOT change, move, round or omit any number. Every figure must "
    "stay attached to the cell line it appears with in the source. "
    "Moving a score from one cell line to another is the worst possible "
    "error.\n"
    "2. Do NOT add any number, fact, claim or biological detail that is "
    "not in the source text.\n"
    "3. Do NOT remove any cell line. Keep every rank heading exactly as "
    "written, on its own line.\n"
    "4. Keep the section headings 'Overview' and 'Actionable Caveats for "
    "Laboratory Validation' exactly as written.\n"
    "5. You may only improve wording, sentence flow and connectives. The "
    "meaning must be identical.\n"
    "6. Do not use markdown headers, bullet points or bold text.\n"
    "7. Output only the rewritten report, nothing before or after it."
)


LABEL_WORDS = (
    "evidence", "similarity", "confidence", "final", "modality",
    "agreement", "subgroup",
)

_PREAMBLE = re.compile(
    r"(?i)^\s*(?:sure[,!.]?\s*)?"
    r"(?:here(?:'s| is)|this is|below is|i(?:'ve| have) rewritten)?"
    r"[^:\n]{0,80}?(?:rewritten|revised|rephrased|version|paragraph)"
    r"[^:\n]{0,40}:\s*")


def _numbers_in(text):
    """
    Every figure in a text, decimals and standalone integers alike.

    Integers matter because the caveat counts ('6 of 10 carry a mutation
    flag') are facts too, and an earlier decimals-only check left them
    unprotected.
    """
    return sorted(re.findall(r'\d+(?:\.\d+)?', text))


def _labelled_numbers(text):
    pairs = []
    for m in re.finditer(r'\d+(?:\.\d+)?', text):
        window = text[max(0, m.start() - 60):m.start()].lower()
        words = re.findall(r'[a-z]+', window)
        label = next((w for w in reversed(words) if w in LABEL_WORDS), "")
        pairs.append((label, m.group()))
    return pairs


def _strip_preamble(text):
    return _PREAMBLE.sub("", text.strip(), count=1).strip()


def _rewrite_is_safe(source, candidate):
    if not candidate:
        return False, "empty"
    if _numbers_in(candidate) != _numbers_in(source):
        return False, "figures changed"
    s_pairs, c_pairs = _labelled_numbers(source), _labelled_numbers(candidate)
    if len(s_pairs) != len(c_pairs):
        return False, "figure count changed"
    for (s_lab, s_num), (c_lab, c_num) in zip(s_pairs, c_pairs):
        if s_num != c_num:
            return False, "figures reordered"
        if s_lab and c_lab and s_lab != c_lab:
            return False, (f"{s_num} was described as {c_lab} "
                           f"rather than {s_lab}")
    if re.search(r"(?i)\b(here is|here's|rewritten|as requested)\b",
                 candidate):
        return False, "conversational framing"
    return True, ""


def _verify_rephrase(source, candidate, df):
    problems = []


    missing = _missing_ranks(candidate, df)
    if missing:
        problems.append(f"missing ranks {missing}")

    src_nums, cand_nums = _numbers_in(source), _numbers_in(candidate)
    if src_nums != cand_nums:
        added = [x for x in cand_nums if x not in src_nums]
        lost = [x for x in src_nums if x not in cand_nums]
        if added:
            problems.append(f"invented figures {added[:5]}")
        if lost:
            problems.append(f"dropped figures {lost[:5]}")

    src_paras = re.split(r'(?m)^Rank \d+ - ', source)[1:]
    cand_paras = re.split(r'(?m)^Rank \d+ - ', candidate)[1:]
    if len(src_paras) == len(cand_paras):
        for i, (sp, cp) in enumerate(zip(src_paras, cand_paras), start=1):
            if set(_numbers_in(sp)) != set(_numbers_in(cp)):
                problems.append(f"figures moved in or out of rank {i}")
                break

    return problems


def _split_sections(report):
    lines = report.split("\n")
    sections, head, body = [], None, []
    for ln in lines:
        is_head = (ln.strip() == "Overview"
                   or ln.startswith("Rank ")
                   or ln.strip().startswith("Actionable Caveats"))
        if is_head:
            if head is not None:
                sections.append((head, "\n".join(body).strip()))
            head, body = ln, []
        elif head is not None:
            body.append(ln)
    if head is not None:
        sections.append((head, "\n".join(body).strip()))
    return sections


PARA_SYSTEM_PROMPT = (
    "Rewrite the paragraph the user gives you so it reads more fluently.\n"
    "RULES:\n"
    "1. Do not change, add, remove or round ANY number. Every figure must "
    "appear exactly as given, attached to the same quantity it describes.\n"
    "2. Do not add any fact, claim or detail that is not already there.\n"
    "3. Keep the same meaning. Only improve wording and sentence flow.\n"
    "4. No markdown, no bullet points, no headings.\n"
    "5. Output ONLY the rewritten paragraph. Do not introduce it, do not "
    "comment on it, and do not write anything before or after it. Never "
    "begin with phrases such as 'Here is the rewritten paragraph' or "
    "'Sure, here you go'. Your first word must be the first word of the "
    "paragraph itself."
)


def generate_explanation(final_top10_df, targetgenelist=None,
                         exclusiongenelist=None, diseasename=None,
                         fusionfilter=False, mutationfilter=False,
                         llm_callable=None, model='llama3.2:3b',
                         host='http://localhost:11434', save=False,
                         use_llm=True):
    try:
        log.info("generate_explanation: starting")
        if final_top10_df is None or final_top10_df.empty:
            raise ValueError("final_top10_df is empty or undefined.")

        df = (final_top10_df
              .copy()
              .sort_values(by='final_score', ascending=False)
              .reset_index(drop=True)
              .head(10))

        draft = render_report(
            df, targetgenelist=targetgenelist,
            exclusiongenelist=exclusiongenelist, diseasename=diseasename,
            fusionfilter=fusionfilter, mutationfilter=mutationfilter)

        if not use_llm:
            explanation = draft
        else:
            caller = llm_callable if llm_callable is not None else ollama_chat
            sections = _split_sections(draft)
            rebuilt, accepted = [], 0

            for head, body in sections:
                best = body
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        if llm_callable is not None:
                            cand = caller(PARA_SYSTEM_PROMPT, body)
                        else:
                            
                            cand = caller(PARA_SYSTEM_PROMPT, body,
                                          model=model, host=host,
                                          seed=SEED + attempt)
                    except Exception as e:
                        log.warning(f"generate_explanation: LLM call failed "
                                    f"on '{head[:24]}' ({e})")
                        break

                    cand = _strip_preamble(cand)
                    ok, why_not = _rewrite_is_safe(body, cand)
                    if ok:
                        best = cand
                        accepted += 1
                        break

                    log.warning(f"generate_explanation: rewrite of "
                                f"'{head[:24]}' rejected on attempt "
                                f"{attempt} - {why_not}")

                rebuilt.append(f"{head}\n{best}")

            explanation = "\n\n".join(rebuilt)
            log.info(f"generate_explanation: {accepted} of {len(sections)} "
                     f"sections used the LLM rewrite")

        if save:
            out_df = pd.DataFrame([{"explanation": explanation}])
            if ut.data_save(out_df, 'test', 'top10',
                            'top10_explanation.csv') != "successfull":
                log.warning("generate_explanation: save did not report "
                            "success.")
        return explanation
    except Exception as e:
        log.error(f"Error generating top-10 explanation: {e}")
        raise