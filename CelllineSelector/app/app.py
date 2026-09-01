import math
import io
import os
import tempfile

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


st.set_page_config(
    page_title="CellLineSelector | AstraZeneca",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

AI_NOTICE = ("This is an AI generated explanation. AI can make mistakes - "
             "please double check the response against the results table.")

P = {
    "bg":           "#F6F1E4",                   
    "bg_glow_a":    "rgba(46, 138, 87, 0.14)",    
    "bg_glow_b":    "rgba(224, 138, 46, 0.12)",   
    "surface":      "rgba(255, 253, 247, 0.92)",  
    "surface_soft": "rgba(255, 253, 247, 0.70)",
    "border":       "#E2D9C3",                    
    "edge":         "rgba(46, 138, 87, 0.28)",    
    "ink":          "#17402A",                    
    "ink_muted":    "#6B7A6A",
    "green_a":      "#3AA968",                    
    "green_b":      "#1E7245",                    
    "green_glow":   "rgba(46, 138, 87, 0.28)",
    "green_soft":   "rgba(46, 138, 87, 0.10)",
    "orange":       "#E08A2E",                    
    "orange_soft":  "rgba(224, 138, 46, 0.12)",
    "chip_ink":     "#1E7245",
    "input_bg":     "#FFFDF7",
    "code_bg":      "#F1EADA",
    "menu_bg":      "#FFFDF7",
    "btn_ink":      "#FFFFFF",
    "panel_bg":     "#FFFDF7",                    
    "shadow":       "0 6px 24px rgba(23, 64, 42, 0.10)",
}

st.markdown(f"""
<style>
    /* page: soft green and orange washes over beige */
    .stApp {{
        background:
            radial-gradient(1100px 520px at 10% -8%,  {P["bg_glow_a"]} 0%, transparent 60%),
            radial-gradient(900px 480px  at 96% 10%,  {P["bg_glow_b"]} 0%, transparent 55%),
            radial-gradient(800px 600px  at 50% 115%, {P["bg_glow_a"]} 0%, transparent 55%),
            {P["bg"]};
        background-attachment: fixed;
    }}

    /* Streamlit's own toolbar (Rerun / Running / menu) */
    header[data-testid="stHeader"] {{ background: transparent; }}
    header[data-testid="stHeader"] *,
    div[data-testid="stToolbar"] *,
    div[data-testid="stStatusWidget"] * {{
        color: {P["ink"]} !important;
        fill: {P["ink"]} !important;
    }}

    /* base text and widget labels */
    .stApp, .stApp p, .stApp li, .stMarkdown,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: {P["ink"]}; }}
    [data-testid="stWidgetLabel"] p,
    .stCheckbox p, .stMultiSelect label, .stSelectbox label {{
        color: {P["ink"]} !important;
    }}
    div[data-testid="stCaptionContainer"] p {{ color: {P["ink_muted"]}; }}
    strong, b {{ color: {P["ink"]}; }}

    /* hero header */
    .az-header {{
        background: {P["surface"]};
        border: 1px solid {P["border"]};
        border-bottom: 3px solid {P["green_a"]};
        box-shadow: {P["shadow"]};
        padding: 1.05rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1.1rem;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .az-header h1 {{
        margin: 0;
        font-size: 1.4rem; font-weight: 800;
        letter-spacing: 0.01em;
        color: {P["green_b"]};
    }}
    .az-header h1 .dot {{ color: {P["orange"]}; }}
    .az-header p {{
        color: {P["ink_muted"]}; font-size: 0.8rem; margin: 0.2rem 0 0 0;
    }}

    .card-title {{
        font-size: 0.95rem; font-weight: 700;
        color: {P["ink"]}; margin-bottom: 0.6rem;
        display: flex; align-items: center; gap: 0.4rem;
    }}

    /* query provenance chips */
    .pill {{
        display: inline-block;
        background: {P["green_soft"]};
        border: 1px solid {P["edge"]};
        color: {P["chip_ink"]};
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.78rem; font-weight: 600;
        margin: 2px 4px 2px 0;
    }}
    .pill-muted {{
        background: {P["orange_soft"]};
        border-color: rgba(224, 138, 46, 0.35);
        color: #9A5B12;
    }}

    .section-label {{
        font-size: 0.72rem; font-weight: 600;
        color: {P["ink_muted"]}; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 4px;
    }}

    /* AI disclosure - orange accent so it reads as a caution */
    .ai-notice {{
        margin-top: 0.5rem;
        padding: 0.55rem 0.85rem;
        border-radius: 8px;
        background: {P["orange_soft"]};
        border: 1px solid rgba(224, 138, 46, 0.35);
        border-left: 3px solid {P["orange"]};
        color: #7A4A10;
        font-size: 0.78rem;
        line-height: 1.45;
    }}

    /* guidance block inside the expander */
    .guide h4 {{
        font-size: 0.9rem; font-weight: 700; color: {P["green_b"]};
        margin: 0.9rem 0 0.35rem 0;
    }}
    .guide p {{
        font-size: 0.83rem; line-height: 1.5; margin: 0.25rem 0;
        color: {P["ink"]};
    }}
    .guide .term {{ font-weight: 700; color: {P["green_b"]}; }}
    .guide .formula {{
        text-align: center;
        font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
        font-size: 0.9rem; font-weight: 700;
        color: {P["green_b"]};
        background: {P["green_soft"]};
        border: 1px solid {P["edge"]};
        border-radius: 8px;
        padding: 0.55rem;
        margin: 0.6rem 0;
    }}

    /* expander */
    div[data-testid="stExpander"] {{
        background: {P["surface_soft"]};
        border: 1px solid {P["border"]};
        border-radius: 12px;
    }}
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p {{
        color: {P["green_b"]} !important;
        font-weight: 700;
    }}
    div[data-testid="stExpander"] svg {{ fill: {P["green_b"]} !important; }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {P["surface_soft"]};
        border: 1px solid {P["border"]};
        border-radius: 12px;
        padding: 5px;
    }}
    .stTabs button[data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        padding: 8px 18px; font-weight: 600;
        color: {P["ink"]} !important;
    }}
    .stTabs button[data-baseweb="tab"] p,
    .stTabs button[data-baseweb="tab"] div {{ color: inherit !important; }}
    .stTabs button[aria-selected="true"] {{
        background: linear-gradient(90deg, {P["green_a"]}, {P["green_b"]}) !important;
        box-shadow: 0 2px 10px {P["green_glow"]};
    }}
    .stTabs button[aria-selected="true"],
    .stTabs button[aria-selected="true"] p,
    .stTabs button[aria-selected="true"] div {{
        color: {P["btn_ink"]} !important;
        -webkit-text-fill-color: {P["btn_ink"]} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ background: transparent !important; }}

    /* metrics */
    div[data-testid="stMetric"] {{
        background: {P["surface"]};
        border: 1px solid {P["border"]};
        border-left: 3px solid {P["green_a"]};
        border-radius: 12px;
        padding: 10px 14px;
        box-shadow: {P["shadow"]};
    }}
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {{ color: {P["ink"]}; }}

    /* query form card */
    div[data-testid="stForm"] {{
        background: {P["surface"]};
        border: 1px solid {P["border"]};
        border-radius: 14px;
        padding: 1.25rem 1.5rem 0.75rem 1.5rem;
        box-shadow: {P["shadow"]};
    }}

    /* select / multiselect */
    div[data-baseweb="select"] > div {{
        background-color: {P["input_bg"]};
        border-color: {P["border"]};
        color: {P["ink"]};
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {{ color: {P["ink"]}; }}
    div[data-baseweb="tag"] {{
        background: linear-gradient(90deg, {P["green_a"]}, {P["green_b"]});
    }}
    div[data-baseweb="tag"],
    div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] div {{
        color: {P["btn_ink"]} !important;
        -webkit-text-fill-color: {P["btn_ink"]} !important;
    }}
    div[data-baseweb="tag"] svg {{ fill: {P["btn_ink"]} !important; }}

    /* dropdown menus render in a portal outside .stApp */
    ul[data-baseweb="menu"] {{
        background-color: {P["menu_bg"]} !important;
        border: 1px solid {P["border"]};
    }}
    ul[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] li div {{ color: {P["ink"]} !important; }}
    ul[data-baseweb="menu"] li:hover {{ background: {P["green_soft"]} !important; }}

    /* explanation panel - light parchment, monospace.
       The textarea is disabled, and browsers grey disabled inputs out via
       -webkit-text-fill-color plus reduced opacity, so both are forced. */
    .stTextArea textarea {{
        background-color: {P["panel_bg"]} !important;
        color: {P["ink"]} !important;
        -webkit-text-fill-color: {P["ink"]} !important;
        opacity: 1 !important;
        border: 1px solid {P["border"]} !important;
        border-left: 3px solid {P["green_a"]} !important;
        border-radius: 10px;
        font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
        font-size: 0.84rem;
        line-height: 1.55;
    }}
    .stTextArea textarea:disabled {{
        color: {P["ink"]} !important;
        -webkit-text-fill-color: {P["ink"]} !important;
        opacity: 1 !important;
    }}

    .stCode, pre, code {{
        background-color: {P["code_bg"]} !important;
        color: {P["ink"]} !important;
        border-radius: 10px;
    }}
    .stCode span {{ color: {P["ink"]} !important; }}

    /* buttons */
    .stButton>button[kind="primary"],
    .stFormSubmitButton>button[kind="primary"],
    .stDownloadButton>button {{
        background: linear-gradient(90deg, {P["green_a"]}, {P["green_b"]});
        border: none;
        font-weight: 700;
        box-shadow: 0 3px 12px {P["green_glow"]};
        transition: box-shadow 0.2s ease, transform 0.15s ease;
    }}
    .stButton>button[kind="primary"],
    .stButton>button[kind="primary"] p,
    .stButton>button[kind="primary"] div,
    .stFormSubmitButton>button[kind="primary"],
    .stFormSubmitButton>button[kind="primary"] p,
    .stFormSubmitButton>button[kind="primary"] div,
    .stDownloadButton>button,
    .stDownloadButton>button p,
    .stDownloadButton>button span,
    .stDownloadButton>button div {{
        color: {P["btn_ink"]} !important;
        -webkit-text-fill-color: {P["btn_ink"]} !important;
    }}
    .stButton>button[kind="primary"]:hover,
    .stFormSubmitButton>button[kind="primary"]:hover,
    .stDownloadButton>button:hover {{
        box-shadow: 0 5px 18px {P["green_glow"]};
        transform: translateY(-1px);
    }}

    .stCheckbox label p {{ color: {P["ink"]}; }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {P["border"]};
        border-radius: 12px;
        box-shadow: {P["shadow"]};
    }}

    hr {{ border-top: 1px solid {P["border"]}; }}

    /* DNA-helix pipeline loader */
    .cls-loading-wrap {{
        display: flex; flex-direction: column; align-items: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        background: {P["surface"]};
        border: 1px solid {P["border"]};
        border-radius: 14px;
        box-shadow: {P["shadow"]};
        margin: 0.8rem 0;
    }}
    .cls-helix {{ display: flex; gap: 13px; height: 64px; align-items: center; }}
    .cls-helix .col {{ position: relative; width: 9px; height: 100%; }}
    .cls-helix .dot {{
        position: absolute; left: 0;
        width: 9px; height: 9px; border-radius: 50%;
        background: {P["green_a"]};
        animation: cls-strand 1.3s ease-in-out infinite;
        animation-delay: calc(var(--i) * -0.13s);
        top: 50%;
    }}
    .cls-helix .dot.b {{
        background: {P["orange"]};
        animation-delay: calc(var(--i) * -0.13s - 0.65s);
        opacity: 0.85;
    }}
    @keyframes cls-strand {{
        0%   {{ transform: translateY(-22px) scale(0.7); }}
        50%  {{ transform: translateY(14px)  scale(1.1); }}
        100% {{ transform: translateY(-22px) scale(0.7); }}
    }}
    .cls-loading-text {{
        margin-top: 1.1rem;
        color: {P["ink"]};
        font-weight: 600; font-size: 0.92rem;
        letter-spacing: 0.02em;
    }}
    .cls-loading-sub {{
        color: {P["ink_muted"]}; font-size: 0.78rem; margin-top: 0.2rem;
    }}
    .cls-loading-text .ell::after {{
        content: '';
        animation: cls-ellipsis 1.4s steps(4, end) infinite;
    }}
    @keyframes cls-ellipsis {{
        0%   {{ content: ''; }}
        25%  {{ content: '.'; }}
        50%  {{ content: '..'; }}
        75%  {{ content: '...'; }}
        100% {{ content: ''; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        .cls-helix .dot {{ animation: none; transform: translateY(0); }}
        .cls-loading-text .ell::after {{ animation: none; content: '...'; }}
        .stButton>button, .stFormSubmitButton>button {{ transition: none; }}
    }}
</style>
""", unsafe_allow_html=True)


def _helix_loader_html(main_text, sub_text):
    cols = "".join(
        f'<div class="col" style="--i:{i}">'
        f'<span class="dot"></span><span class="dot b"></span>'
        f'</div>'
        for i in range(9))
    return (
        f'<div class="cls-loading-wrap">'
        f'  <div class="cls-helix">{cols}</div>'
        f'  <div class="cls-loading-text">{main_text}<span class="ell"></span></div>'
        f'  <div class="cls-loading-sub">{sub_text}</div>'
        f'</div>')


st.markdown("""
<div class="az-header">
    <div>
        <h1>AstraZeneca <span class="dot">&middot;</span> CellLineSelector</h1>
        <p>Multi-omics cell line selection &amp; ranking</p>
    </div>
</div>
""", unsafe_allow_html=True)

GUIDE_HTML = f"""
<div class="guide">
<p>CellLineSelector recommends the most suitable cancer cell lines for
studying a chosen target gene, drawn from a reference set of 1,479
characterised cell lines. It ranks each line by combining multi-omics
expression evidence, molecular-profile similarity and data confidence
into a single transparent score, and surfaces mutation and fusion flags
so recommendations stay auditable.</p>

<h4>Query fields</h4>
<p><span class="term">Target gene(s)</span> &mdash; required, at least one.
The gene or genes you want the recommended cell lines to express.</p>
<p><span class="term">Exclude gene(s)</span> &mdash; optional. Cell lines
that strongly express these genes are down-ranked. A gene cannot be both
a target and an exclusion.</p>
<p><span class="term">Disease</span> &mdash; optional. Restricts results to
one disease category. Scores are unchanged; non-matching lines are
hidden, so a filtered line keeps the score it would have had in an
unfiltered query.</p>
<p><span class="term">Mutation / Fusion filter</span> &mdash; three settings
each: <em>show flag</em> (default, ranking unaffected), <em>exclude
flagged</em>, or <em>require flagged</em>.</p>

<h4>How the scores are calculated</h4>
<p>Every cell line receives a <span class="term">Final Score</span> between
0 and 1, formed by multiplying three independent components. Because they
are multiplied, a line must do well on <strong>all three</strong>; a low
value on any one strongly suppresses the result.</p>
<p><span class="term">Evidence</span> &mdash; how strongly the target
gene(s) are expressed, averaged across DepMap RNA, HPA RNA and
mass-spectrometry proteomics. A missing source contributes zero, so lines
with incomplete data score lower.</p>
<p><span class="term">Similarity</span> &mdash; how closely the line's
overall multi-omics profile matches the target gene's molecular
signature, measured by cosine similarity in the MOFA factor space. This
captures biological context beyond raw expression.</p>
<p><span class="term">Confidence</span> &mdash; how complete and consistent
the underlying data is, combining how many of the three sources are
present with how well they agree.</p>
<div class="formula">Final Score = Evidence &times; Similarity &times; Confidence</div>
<p>Exclusion genes, if given, apply a penalty to both Evidence and
Similarity. Cell lines are ranked in descending order of the final score
and the top 10 are shown.</p>
<p><span class="term">Fusion / Mutation flags</span> &mdash; shown for
context only; they do <strong>not</strong> change the score or the
ranking. A flag means the line carries a gene fusion or somatic mutation
in the queried gene &mdash; whether that is desirable is left to the
researcher.</p>
</div>
"""


FILTER_LABELS = {
    3: "Default - show flag, no filtering",
    1: "No - exclude flagged cell lines",
    2: "Yes - require flagged cell lines",
}
FILTER_CODE_BY_LABEL = {v: k for k, v in FILTER_LABELS.items()}
FILTER_ORDER = [3, 1, 2]


def is_flagged(val):
    if val is None:
        return False
    return str(val).strip().lower() not in (
        'false', 'no', '0', 'nan', 'none', '')


def flag_count(df, col):
    if df is None or col not in df.columns:
        return 0
    def _count(val):
        if val is None:
            return False
        try:
            return float(str(val)) > 0
        except (ValueError, TypeError):
            return str(val).strip().lower() not in (
                'false', 'no', '0', 'nan', 'none', '')
    return int(df[col].apply(_count).sum())


def safe_float(val):
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def pdf_text(s):

    if s is None:
        return ""
    s = str(s)
    replacements = {
        "\u2014": "-", "\u2013": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...",
        "\u2022": "-",
        "\u00a0": " ",
    }
    for u, a in replacements.items():
        s = s.replace(u, a)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def show_plot(plot_obj):
    if plot_obj is None:
        st.caption("No cluster plot available for this query.")
        return
    if isinstance(plot_obj, str):
        p = Path(plot_obj)
        if p.exists():
            st.image(str(p))
        else:
            st.caption(f"Plot file not found: {plot_obj}")
    else:
        try:
            st.pyplot(plot_obj, clear_figure=False)
        except Exception as e:
            st.caption(f"Could not display plot: {e}")


def gene_evidence_columns(df):
 
    if df is None:
        return []
    return [c for c in df.columns
            if c.startswith("ENSG") and (c.endswith("_x") or c.endswith("_y"))]


TOP10_BASE_COLUMNS = [
    "cell_line_name", "stripped_cell_line_name", "ModelID", "CVCL_ID",
    "CCLE_Name", "primary_disease", "lineage",
    "net_evidence", "confidence_score", "net_similarity", "final_score",
    "fusion_flag", "mutation_flag",
]

ALL_DATA_FIXED_COLUMNS = [
    "cell_line_name", "stripped_cell_line_name", "ModelID", "CVCL_ID",
    "CCLE_Name", "primary_disease", "lineage",
]
ALL_DATA_TRAILING_COLUMNS = [
    "net_evidence", "confidence_score", "net_similarity", "final_score",
    "fusion_flag", "fusion_type", "mutation_flag", "mutation_type",
]

COLUMN_LABELS = {
    "cell_line_name": "Cell Line",
    "stripped_cell_line_name": "Stripped Name",
    "ModelID": "ACH ID",
    "CVCL_ID": "CVCL ID",
    "CCLE_Name": "CCLE Name",
    "primary_disease": "Disease",
    "lineage": "Lineage",
    "net_evidence": "Evidence",
    "confidence_score": "Confidence",
    "net_similarity": "Similarity",
    "final_score": "Final Score",
    "fusion_flag": "Fusion Flag",
    "fusion_type": "Fusion Type",
    "mutation_flag": "Mutation Flag",
    "mutation_type": "Mutation Type",
}


def display_dataframe(df, columns=None, height=None):
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    cols = [c for c in (columns or df.columns) if c in df.columns]
    view = df[cols].rename(columns=COLUMN_LABELS)
    if height is None:
        st.dataframe(view, use_container_width=True)
    else:
        st.dataframe(view, use_container_width=True, height=height)


def pill(text, muted=False):
    cls = "pill pill-muted" if muted else "pill"
    return f'<span class="{cls}">{text}</span>'



@st.cache_data
def load_gene_list():
    p = _REPO_ROOT / 'data/lookup/gene_maps/gene_ensg_map.csv'
    if p.exists():
        df = pd.read_csv(p)
        return sorted(df['gene_symbol'].dropna().unique().tolist())
    return []


@st.cache_data
def load_disease_list():
    p = _REPO_ROOT / 'data/lookup/master/master_lookup.csv'
    if p.exists():
        df = pd.read_csv(p)
        return sorted(df['primary_disease'].dropna().unique().tolist())
    return []


gene_list    = load_gene_list()
disease_list = load_disease_list()


def run_pipeline(target_genes, exclusion_genes, disease,
                 fusion_filter_code, mutation_filter_code):
    from app.applicationrunner import pipelinerun
    return pipelinerun(
        targetgenelist=target_genes,
        exclusiongenelist=exclusion_genes if exclusion_genes else [],
        diseasename=disease if disease and disease != 'All' else None,
        fusionfilter=fusion_filter_code,
        mutationfilter=mutation_filter_code,
    )

if 'results' not in st.session_state:
    st.session_state.results = None
if 'search_error' not in st.session_state:
    st.session_state.search_error = None

R = st.session_state.results
show_mutation_tab = bool(
    R and flag_count(R.get('top10'), 'mutation_flag') > 0
    and R.get('mut_filter_code') != 1
)
show_fusion_tab = bool(
    R and flag_count(R.get('top10'), 'fusion_flag') > 0
    and R.get('fus_filter_code') != 1
)

tab_labels = ["Search & Results", "All Data", "Cluster Plot", "Export"]
insert_at = 2
if show_fusion_tab:
    tab_labels.insert(insert_at, "Fusion Reference")
if show_mutation_tab:
    tab_labels.insert(insert_at, "Mutation Reference")

tabs = st.tabs(tab_labels)
tab_map = dict(zip(tab_labels, tabs))

tab_search  = tab_map["Search & Results"]
tab_all     = tab_map["All Data"]
tab_cluster = tab_map["Cluster Plot"]
tab_export  = tab_map["Export"]
tab_mut_ref = tab_map.get("Mutation Reference")
tab_fus_ref = tab_map.get("Fusion Reference")


with tab_search:

    with st.expander("About CellLineSelector - query fields and scoring",
                     expanded=False):
        st.markdown(GUIDE_HTML, unsafe_allow_html=True)

    with st.form("search_form", clear_on_submit=False):
        st.markdown('<div class="card-title">Query</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Target Gene(s)** :red[*required*]")
            target_genes = st.multiselect(
                "Target gene(s)", options=gene_list,
                placeholder="Type or select genes...",
                label_visibility="collapsed")
        with c2:
            st.markdown("**Exclude Gene(s)** &nbsp;&middot;&nbsp; optional")
            exclusion_genes = st.multiselect(
                "Exclude gene(s)", options=gene_list,
                placeholder="Type or select genes...",
                label_visibility="collapsed")

        st.markdown("")
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("**Disease**")
            disease = st.selectbox(
                "Disease",
                options=['All'] + disease_list, index=0,
                label_visibility="collapsed")
        with f2:
            st.markdown("**Mutation Filter**")
            mutation_label = st.selectbox(
                "Mutation filter",
                options=[FILTER_LABELS[c] for c in FILTER_ORDER],
                index=0, label_visibility="collapsed",
                help="Default: flag shown, not enforced. 'No' removes "
                     "flagged cell lines; 'Yes' keeps only flagged ones.")
        with f3:
            st.markdown("**Fusion Filter**")
            fusion_label = st.selectbox(
                "Fusion filter",
                options=[FILTER_LABELS[c] for c in FILTER_ORDER],
                index=0, label_visibility="collapsed",
                help="Default: flag shown, not enforced. 'No' removes "
                     "flagged cell lines; 'Yes' keeps only flagged ones.")

        st.markdown("")
        b1, b2, b3 = st.columns([1, 1, 1])
        with b2:
            search_btn = st.form_submit_button(
                "Search", type="primary", use_container_width=True)

    if search_btn:
        overlap = set(target_genes) & set(exclusion_genes)
        if not target_genes:
            st.session_state.search_error = (
                "Add at least one target gene before searching.")
        elif overlap:
            st.session_state.search_error = (
                "A gene cannot be both a target and an exclusion gene: "
                + ", ".join(sorted(overlap)) + ". Remove it from one list.")
        else:
            st.session_state.search_error = None
            mutation_filter_code = FILTER_CODE_BY_LABEL[mutation_label]
            fusion_filter_code   = FILTER_CODE_BY_LABEL[fusion_label]

            loader_slot = st.empty()
            loader_slot.markdown(
                _helix_loader_html(
                    "Analysing multi-omics space",
                    "MOFA factors - cosine similarity - scoring - LLM narrative"),
                unsafe_allow_html=True)
            try:
                (top10_df, explanation, metabolomnic_df,
                 fusion_reference_df, mutation_reference_df,
                 final_mutated_all_df, kmeansplot) = run_pipeline(
                    target_genes, exclusion_genes,
                    disease if disease != 'All' else None,
                    fusion_filter_code, mutation_filter_code)

                st.session_state.results = {
                    'top10'          : top10_df,
                    'all_data'       : final_mutated_all_df,
                    'metabolomic'    : metabolomnic_df,
                    'fusion_ref'     : fusion_reference_df,
                    'mutation_ref'   : mutation_reference_df,
                    'kmeans_plot'    : kmeansplot,
                    'explanation'    : explanation,
                    'target_genes'   : target_genes,
                    'excl_genes'     : exclusion_genes,
                    'disease'        : disease,
                    'mut_filter_code': mutation_filter_code,
                    'fus_filter_code': fusion_filter_code,
                }
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                loader_slot.empty()

    if st.session_state.search_error:
        st.error(st.session_state.search_error)

    R = st.session_state.results

    if R is None:
        st.markdown(f"""
        <div style="text-align:center;padding:2.5rem 2rem;color:{P["ink_muted"]};">
            <div style="font-size:1.05rem;font-weight:600;
                        color:{P["ink"]};">
                Select a target gene and click Search
            </div>
            <div style="font-size:0.85rem;margin-top:0.4rem;">
                Results will appear here
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        top10_df = R['top10']

        st.markdown("")
        chips = [pill(f"{g}") for g in R['target_genes']]
        chips += [pill(f"{g}") for g in R['excl_genes']]
        if R['disease'] and R['disease'] != 'All':
            chips.append(pill(f"{R['disease']}"))
        chips.append(pill(
            f"Mutation: {FILTER_LABELS[R['mut_filter_code']].split(' - ')[0]}",
            muted=True))
        chips.append(pill(
            f"Fusion: {FILTER_LABELS[R['fus_filter_code']].split(' - ')[0]}",
            muted=True))
        st.markdown(" ".join(chips), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="card-title">Top 10 Cell Lines</div>',
                    unsafe_allow_html=True)
        display_dataframe(top10_df, columns=TOP10_BASE_COLUMNS)

        st.markdown("---")
        st.markdown('<div class="card-title">Explanation</div>',
                    unsafe_allow_html=True)
        expl_text = str(R.get('explanation') or '').strip()
        if expl_text:
            st.text_area(
                label="explanation", value=expl_text, height=320,
                label_visibility="collapsed", disabled=True)
            st.markdown(f'<div class="ai-notice">{AI_NOTICE}</div>',
                        unsafe_allow_html=True)
        else:
         
            st.info("No explanation was returned for this query.")


═
if tab_mut_ref is not None:
    with tab_mut_ref:
        st.markdown('<div class="card-title">Mutation Reference</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Full variant-level annotation for the mutation-flagged "
            "cell lines in the top-10 result.")
        display_dataframe(R.get('mutation_ref'))

if tab_fus_ref is not None:
    with tab_fus_ref:
        st.markdown('<div class="card-title">Fusion Reference</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Gene fusion partner detail for the fusion-flagged "
            "cell lines in the top-10 result.")
        display_dataframe(R.get('fusion_ref'))


with tab_all:
    st.markdown('<div class="card-title">Full scored &amp; filtered cell line list</div>',
                unsafe_allow_html=True)
    if R is None:
        st.info("Run a search first.")
    else:
        all_df = R.get('all_data')
        st.caption(
            f"{0 if all_df is None else len(all_df)} cell line(s) after "
            f"disease / mutation / fusion filtering (pre top-10 cut-off).")
        display_dataframe(
            all_df,
            columns=ALL_DATA_FIXED_COLUMNS + gene_evidence_columns(all_df)
                    + ALL_DATA_TRAILING_COLUMNS,
            height=560,
        )

with tab_cluster:
    st.markdown('<div class="card-title">KMeans Cluster Plot</div>',
                unsafe_allow_html=True)
    st.caption(
        "Top-10 recommended cell lines highlighted within their "
        "biological sub-group in the MOFA factor space.")
    if R is None:
        st.info("Run a search first.")
    else:
        plot_col, _ = st.columns([3, 1])
        with plot_col:
            show_plot(R.get('kmeans_plot'))


with tab_export:
    st.markdown('<div class="card-title">Export Report</div>',
                unsafe_allow_html=True)

    if R is None:
        st.info("Run a search first.")
    else:
        top10_df   = R.get('top10')
        all_df     = R.get('all_data')
        mut_ref_df = R.get('mutation_ref')
        fus_ref_df = R.get('fusion_ref')
        expl_text  = str(R.get('explanation') or '').strip()
        kmeans_plot = R.get('kmeans_plot')

        left, right = st.columns([1.3, 1])

        with left:
            st.markdown("**Choose what to include**")
            e1, e2 = st.columns(2)
            with e1:
                inc_top10 = st.checkbox("Top 10 cell lines", value=True)
                inc_expl  = st.checkbox("Explanation", value=True)
                inc_mut = st.checkbox(
                    "Mutation reference", value=False,
                    disabled=mut_ref_df is None or (hasattr(mut_ref_df, "empty") and mut_ref_df.empty),
                    help="Unavailable - no mutation-flagged cell lines."
                         if (mut_ref_df is None or (hasattr(mut_ref_df, "empty") and mut_ref_df.empty)) else None)
            with e2:
                inc_all   = st.checkbox("All data (full table)", value=False)
                inc_kmeans = st.checkbox(
                    "Cluster plot", value=False,
                    disabled=kmeans_plot is None,
                    help="Unavailable - no cluster plot for this query."
                         if kmeans_plot is None else None)
                inc_fus = st.checkbox(
                    "Fusion reference", value=False,
                    disabled=fus_ref_df is None or (hasattr(fus_ref_df, "empty") and fus_ref_df.empty),
                    help="Unavailable - no fusion-flagged cell lines."
                         if (fus_ref_df is None or (hasattr(fus_ref_df, "empty") and fus_ref_df.empty)) else None)

        with right:
            st.markdown("**Query summary**")
            st.code(
                f"Target gene(s)   : {', '.join(R['target_genes'])}\n"
                f"Exclude gene(s)  : {', '.join(R['excl_genes']) or 'None'}\n"
                f"Disease filter   : {R['disease'] or 'All'}\n"
                f"Mutation filter  : {FILTER_LABELS[R['mut_filter_code']]}\n"
                f"Fusion filter    : {FILTER_LABELS[R['fus_filter_code']]}",
                language=None)

        nothing_selected = not any(
            [inc_top10, inc_all, inc_expl, inc_kmeans, inc_mut, inc_fus])

        def build_pdf():
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            def header_band():
                pdf.set_fill_color(30, 114, 69)
                pdf.set_text_color(255, 255, 255)
                pdf.rect(0, 0, pdf.w, 22, 'F')
                pdf.set_font('Helvetica', 'B', 14)
                pdf.set_xy(10, 6)
                pdf.cell(0, 10, pdf_text('AstraZeneca | CellLineSelector Report'))
                pdf.set_text_color(0, 0, 0)

            def _fit(text, width):
                text = pdf_text(text)
                if not text:
                    return ''
                if pdf.get_string_width(text) <= width - 1:
                    return text
                while text and pdf.get_string_width(text + '..') > width - 1:
                    text = text[:-1]
                return (text + '..') if text else ''

            def _plot_png_bytes(obj):
                try:
                    if obj is None:
                        return None
                    if isinstance(obj, (bytes, bytearray)):
                        return bytes(obj)
                    if isinstance(obj, str):
                        p = Path(obj)
                        return p.read_bytes() if p.exists() else None
                    buf = io.BytesIO()
                    obj.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    data = buf.getvalue()
                    return data if data else None
                except Exception:
                    return None

            def render_table(df, title, columns=None):
                if df is None or (hasattr(df, 'empty') and df.empty):
                    return
                cols = [c for c in (columns or df.columns) if c in df.columns]
                landscape = len(cols) > 8
                pdf.add_page(orientation='L' if landscape else 'P')
                header_band()
                pdf.set_xy(10, 28)
                usable = pdf.w - 20
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, pdf_text(title))
                pdf.ln(8)
                col_w = usable / max(len(cols), 1)
                pdf.set_fill_color(226, 240, 231)
                pdf.set_font('Helvetica', 'B', 6.5)
                pdf.set_x(10)
                for c in cols:
                    pdf.cell(col_w, 7, _fit(COLUMN_LABELS.get(c, c), col_w),
                             border=1, fill=True)
                pdf.ln()
                pdf.set_font('Helvetica', '', 6.5)
                for _, row in df.iterrows():
                    pdf.set_x(10)
                    for c in cols:
                        v = row.get(c, '')
                        fv = safe_float(v)
                        text = f"{fv:.3f}" if (fv is not None and c not in
                                               ('ModelID', 'CVCL_ID')) else str(v)
                        pdf.cell(col_w, 6, _fit(text, col_w), border=1)
                    pdf.ln()

            def render_text_section(title, body, note=None):
                pdf.add_page(orientation='P')
                header_band()
                pdf.set_xy(10, 28)
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, pdf_text(title))
                pdf.ln(8)
                pdf.set_font('Helvetica', '', 8)
                pdf.set_x(10)
                pdf.multi_cell(0, 5, pdf_text(body))
                if note:
                    pdf.ln(2)
                    pdf.set_font('Helvetica', 'I', 7)
                    pdf.set_x(10)
                    pdf.multi_cell(0, 4, pdf_text(note))

            def render_plot():
                pdf.add_page(orientation='P')
                header_band()
                pdf.set_xy(10, 28)
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, 'Cluster Plot')
                pdf.ln(8)
                png = _plot_png_bytes(kmeans_plot)
                if png:
                    tmp = os.path.join(tempfile.gettempdir(), 'cls_export_plot.png')
                    with open(tmp, 'wb') as fh:
                        fh.write(png)
                    pdf.image(tmp, x=10, w=min(pdf.w - 20, 190))
                else:
                    pdf.set_font('Helvetica', 'I', 8)
                    pdf.multi_cell(0, 5,
                                   '[Cluster plot could not be rendered for this query.]')

            def _safe(section_fn, label):
                try:
                    section_fn()
                except Exception as se:
                    pdf.set_font('Helvetica', 'I', 8)
                    pdf.multi_cell(0, 5, pdf_text(f"[{label} could not be added: {se}]"))
                    pdf.ln(2)


            pdf.add_page(orientation='P')
            header_band()
            pdf.set_xy(10, 28)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 6, 'Query Summary')
            pdf.ln(7)
            pdf.set_font('Helvetica', '', 9)
            for line in [
                f"Target gene(s)   : {', '.join(R['target_genes'])}",
                f"Exclude gene(s)  : {', '.join(R['excl_genes']) or 'None'}",
                f"Disease filter   : {R['disease'] or 'All'}",
                f"Mutation filter  : {FILTER_LABELS[R['mut_filter_code']]}",
                f"Fusion filter    : {FILTER_LABELS[R['fus_filter_code']]}",
            ]:
                pdf.set_x(10)
                pdf.cell(0, 5, pdf_text(line))
                pdf.ln(5)

            if inc_top10:
                _safe(lambda: render_table(top10_df, 'Top 10 Cell Lines',
                                           TOP10_BASE_COLUMNS),
                      'Top 10 table')
            if inc_expl and expl_text:
                _safe(lambda: render_text_section('Explanation', expl_text, AI_NOTICE),
                      'Explanation')
            if inc_kmeans and kmeans_plot is not None:
                _safe(render_plot, 'Cluster plot')
            if inc_fus:
                _safe(lambda: render_table(fus_ref_df, 'Fusion Reference'),
                      'Fusion Reference')
            if inc_mut:
                _safe(lambda: render_table(mut_ref_df, 'Mutation Reference'),
                      'Mutation Reference')
            if inc_all:
                _safe(lambda: render_table(
                        all_df, 'All Data',
                        ALL_DATA_FIXED_COLUMNS + gene_evidence_columns(all_df)
                        + ALL_DATA_TRAILING_COLUMNS),
                      'All Data table')

            def _to_bytes(o):
                if not o:
                    return b''
                if isinstance(o, (bytes, bytearray)):
                    return bytes(o)
                return o.encode('latin-1')

            data = _to_bytes(pdf.output())
            if not data:
                try:
                    data = _to_bytes(pdf.output(dest='S'))
                except Exception:
                    data = b''
            return data

        st.markdown("---")
        if not PDF_AVAILABLE:
            st.warning("fpdf2 not installed. Run: pip install fpdf2")
        elif nothing_selected:
            st.warning("Select at least one item above to build a PDF.")
        else:
            gene_str = '_'.join(R['target_genes'])
            try:
                pdf_data = build_pdf()
            except Exception as e:
                import traceback
                st.error(f"PDF build failed: {e}")
                st.code(traceback.format_exc())
                pdf_data = None
            if pdf_data:
                dl_col, _ = st.columns([1, 2])
                with dl_col:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"CellLineSelector_{gene_str}.pdf",
                        mime="application/pdf",
                        use_container_width=True)
            else:
                st.error("PDF build produced no data.")
