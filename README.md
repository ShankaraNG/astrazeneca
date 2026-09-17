# CellLineSelector

A transparent multi-omics framework for cancer cell line recommendation. Given one or more target genes, CellLineSelector ranks 1,479 DepMap cancer cell lines by suitability for studying those genes, combining gene-expression evidence, multi-omics similarity, and data confidence into a single auditable score, with optional exclusion genes, disease filtering, and mutation/fusion flagging. A locally hosted LLM generates a natural-language rationale for each recommendation.

## Live End Point

The application has been deployed on the AWS services on the below URL

http://celllineselectorlb-450115590.us-east-1.elb.amazonaws.com/

## Architecture

The system runs in two phases:

1. **`ml_build/` — offline pipeline.** Run once to clean the raw data and build all machine-learning models and scoring artifacts (MOFA factors, cosine-similarity matrix, KMeans clustering, lookup tables, and scaled scoring tables). Computationally heavy; run when the underlying data or models change.
2. **`app/` — online application.** Loads the pre-built artifacts and answers researcher queries in near real time. Query-time work reduces to table look-ups, vector averaging, scoring, and the LLM call. Available as a Streamlit UI (`app/app.py`) and a FastAPI service (`app/main.py`).

## Repository layout

```
CelllineSelector/
├── ml_build/                 # Offline pipeline (build models + artifacts)
│   ├── main.py               # Entry point for the offline build
│   ├── pipelinerunner.py     # Orchestrates the sequential build stages
│   ├── datacleaner.py        # Raw data cleaning / alignment
│   ├── preprocessing.py      # Z-score and MinMax normalisation
│   ├── lookuptable.py        # Builds master / gene-ENSG / gene-protein lookups
│   ├── training.py           # MOFA + KMeans (and KNN, model-selection only)
│   ├── cosinematrix.py       # Cosine-similarity matrix construction
│   ├── testing.py            # Model-selection / evaluation
│   ├── utils.py
│   └── logger.py
│
├── app/                      # Online application (serve queries)
│   ├── app.py                # Streamlit user interface
│   ├── main.py               # FastAPI service (POST /recommend)
│   ├── applicationrunner.py  # pipelinerun(): query orchestration
│   ├── modelloader.py        # Loads pre-built artifacts
│   ├── preprocessing.py      # Query-time data retrieval
│   ├── scoring.py            # Evidence / similarity / confidence scoring
│   ├── mutation.py           # Mutation flagging
│   ├── fusion.py             # Fusion flagging
│   ├── aimodel.py            # LLM explanation layer (Ollama / Llama 3.2:3b)
│   └── logger.py
│
├── models/                   # Trained models (.joblib) — built by ml_build
├── artifacts/                # Model-selection plots and metrics
├── images/                   # Per-query cluster plots
├── logs/                     # application.log, machinelearning.log
└── requirements.txt
```

## Prerequisites

- **Python 3.10 or 3.11** (i.e. **< 3.12** — see the version note below).
- **[Ollama](https://ollama.com/)** installed and running locally, with the Llama 3.2 3B model pulled, for the LLM explanation layer (see step 5 below).
- The raw DepMap / HPA / CCLE data files in the location the pipeline expects.
  **[CHECK] State the expected raw-data path here — the pipeline reads from a data directory that isn't in the repo.**

## Environment setup

### 1. Python version

Use **Python 3.10 or 3.11** (i.e. **< 3.12**). Some of the scientific
dependencies (e.g. `mofapy2` and the pinned `numpy`/`scikit-learn` builds)
are not yet compatible with Python 3.12+, so a 3.12 interpreter will fail to
resolve or install the requirements.

Check your version first:

```bash
python --version
```

If your default `python` is 3.12 or newer, install 3.11 alongside it (from
[python.org](https://www.python.org/downloads/) or via `pyenv`) and use that
interpreter explicitly when creating the environment below (e.g.
`py -3.11 -m venv .venv` on Windows, or `python3.11 -m venv .venv` on
macOS/Linux).

### 2. Create the virtual environment

From the `CelllineSelector/` directory:

```bash
python -m venv .venv
```

This creates an isolated `.venv/` folder so the project's packages don't
interfere with your system Python.

### 3. Activate the environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash / MINGW64):**
```bash
source .venv/Scripts/activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Once active, your prompt is prefixed with `(.venv)`. Run every command in this
README from inside the activated environment. To leave it later, run
`deactivate`.

> **PowerShell note:** if activation is blocked by an execution-policy error,
> run once in that PowerShell window:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 4. Install dependencies

With the environment active, upgrade `pip` and install everything from
`requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs all Python dependencies pinned for the project. Verify the
install completed without errors before running the pipeline or the app.

### 5. Install and run Ollama (LLM explanation layer)

The natural-language rationale for each recommendation is generated by a
**locally hosted** Llama 3.2 3B model served through Ollama, so no query data
ever leaves the machine. Ollama must be installed and running for the app's
explanation layer to work.

1. **Install Ollama** from [ollama.com/download](https://ollama.com/download)
   (Windows, macOS, and Linux installers are provided). On Linux you can
   instead run:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start the Ollama service.** On Windows and macOS the desktop app starts
   it automatically once installed; on Linux (or to run it manually) use:
   ```bash
   ollama serve
   ```
   This serves the local API on `http://localhost:11434`.

3. **Pull the model** (one-off download, a few GB):
   ```bash
   ollama pull llama3.2:3b
   ```

4. **Verify** it is available:
   ```bash
   ollama list
   ```
   `llama3.2:3b` should appear in the list.

> Keep Ollama running whenever you use the app. If the Ollama endpoint is
> unavailable at query time, the app falls back to the deterministic report
> without the rephrased explanation, so recommendations still work — but for
> the full experience Ollama should be up.

## Running the offline pipeline (`ml_build`)

Run this **first**, and whenever the source data or models change. It cleans the data, trains MOFA/KMeans, builds the cosine matrix, and writes the lookup and scoring artifacts into `models/` and the scoring/lookup directories.

```bash
# from the CelllineSelector/ directory
python -m ml_build.main
```

This is a one-off, long-running build (order of ~30 minutes, dominated by the offline cosine-similarity computation). It must complete successfully before the app can serve queries, since the app only loads pre-built artifacts.

## Running the online application (`app`)

The app assumes the `ml_build` step has already produced the artifacts. Run both commands from the `CelllineSelector/` directory with the virtual environment active.

### Streamlit interface (primary UI)

```bash
streamlit run app/app.py
```

This opens the single-page UI with the query form and result tabs: recommendations, full results, cluster view, and mutation/fusion references.

### FastAPI service

```bash
uvicorn app.main:app
```

- Interactive API docs: `http://127.0.0.1:8000/docs`
- Health check: `GET http://127.0.0.1:8000/health`

**Example request** — `POST http://127.0.0.1:8000/recommend`:

```json
{
  "targetgenelist": ["EGFR"],
  "exclusiongenelist": null,
  "diseasename": null,
  "fusionfilter": 0,
  "mutationfilter": 0
}
```

Field notes: `targetgenelist` is required (at least one gene); `fusionfilter` and `mutationfilter` are integers — `0` = off, `1` = remove flagged lines, `2` = require flagged lines; `diseasename` must match a `primary_disease` category or be `null` for all lineages.

Add `--reload` during development (`uvicorn app.main:app --reload`) to auto-restart on code changes.

## Logs

- `logs/machinelearning.log` — offline pipeline
- `logs/application.log` — online application
