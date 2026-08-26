import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import app.applicationrunner as runner
from app.logger import get_logger

log = get_logger('API')

app = FastAPI(
    title="CellLineSelector API",
    description="Multi-omics cell line recommendation with an LLM-generated rationale.",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    targetgenelist: List[str] = Field(..., min_items=1, description="Target gene symbols (want expressed/relevant).")
    exclusiongenelist: Optional[List[str]] = Field(default=None, description="Exclusion gene symbols (want absent/avoided).")
    diseasename: Optional[str] = Field(default=None, description="Disease restriction, or null for all lineages.")
    fusionfilter: int = Field(default=0, ge=0, le=2, description="0 = off, 1 = remove fusion-positive, 2 = require fusion-positive.")
    mutationfilter: int = Field(default=0, ge=0, le=2, description="0 = off, 1 = remove mutation-positive, 2 = require mutation-positive.")


def frame_to_records(df):
    try:
        if df is None:
            return []
        records = df.to_dict(orient='records')
        clean = []
        for rec in records:
            out = {}
            for k, v in rec.items():
                if isinstance(v, float) and not np.isfinite(v):
                    out[k] = None
                elif isinstance(v, np.integer):
                    out[k] = int(v)
                elif isinstance(v, np.floating):
                    fv = float(v)
                    out[k] = fv if np.isfinite(fv) else None
                elif pd.isna(v):
                    out[k] = None
                else:
                    out[k] = v
            clean.append(out)
        return clean
    except Exception as e:
        log.error(f"Error converting dataframe to records: {e}")
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        log.info(f"/recommend: targets={request.targetgenelist} "
                 f"exclusions={request.exclusiongenelist} disease={request.diseasename} "
                 f"fusionfilter={request.fusionfilter} mutationfilter={request.mutationfilter}")
        final_top10_df, explanation, metabolomnic_df, fusion_reference_df, mutation_reference_df, final_mutated_df, kmeansplotforselectedcell = runner.pipelinerun(request.targetgenelist, request.exclusiongenelist, request.diseasename, request.fusionfilter, request.mutationfilter )

        response = {
            "query": {
                "targetgenelist": request.targetgenelist,
                "exclusiongenelist": request.exclusiongenelist,
                "diseasename": request.diseasename,
                "fusionfilter": request.fusionfilter,
                "mutationfilter": request.mutationfilter
            },
            "top10": frame_to_records(final_top10_df),
            "explanation": explanation,
        }
        log.info(f"/recommend: returning {len(response['top10'])} cell line(s)")
        return response

    except ValueError as e:
        log.error(f"/recommend: bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"/recommend: pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")