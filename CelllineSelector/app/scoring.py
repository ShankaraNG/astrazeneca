import pandas as pd
import numpy as np
from app.logger import get_logger


log = get_logger('Scoring')

LAMBDA_VAL = 1.5


def calculateevidencescore(data_df, type):
    try:
        log.info(f"calculateevidencescore (type={type}): input shape {data_df.shape}")
        if 'ModelID' not in data_df.columns:
            raise KeyError("data_df must contain a 'ModelID' column")
        gene_cols = [c for c in data_df.columns if c != 'ModelID']
        log.info(f"calculateevidencescore: {len(gene_cols)} value column(s)")
        data_df[gene_cols] = data_df[gene_cols].fillna(0)

        if type == "target":
            data_df['target_evidence'] = data_df[gene_cols].mean(axis=1)
            log.info(f"target_evidence: min={data_df['target_evidence'].min():.4f} "
                     f"max={data_df['target_evidence'].max():.4f}")
            return data_df[['ModelID', 'target_evidence']]
        elif type == "exclusion":
            data_df['exclusion_evidence'] = data_df[gene_cols].mean(axis=1)
            data_df['exclusion_percentile'] = data_df['exclusion_evidence'].rank(pct=True)
            data_df['exclusion_penalty'] = np.exp(-LAMBDA_VAL * data_df['exclusion_percentile'])
            log.info(f"exclusion_penalty: min={data_df['exclusion_penalty'].min():.4f} "
                     f"max={data_df['exclusion_penalty'].max():.4f}")
            return data_df[['ModelID', 'exclusion_evidence', 'exclusion_percentile', 'exclusion_penalty']]
        else:
            raise ValueError(f"type must be 'target' or 'exclusion', got {type!r}")
    except Exception as e:
        log.error(f"Error in calculateevidencescore (type={type}): {e}")
        raise


def calculateconfidencescore(data_df):
    try:
        log.info(f"calculateconfidencescore: input shape {data_df.shape}")
        if 'ModelID' not in data_df.columns:
            raise KeyError("data_df must contain a 'ModelID' column")
        gene_cols = [c for c in data_df.columns if c != 'ModelID']
        log.info(f"calculateconfidencescore: pooling {len(gene_cols)} value column(s)")
        raw_features_df = data_df[gene_cols].copy()
        copied_df = data_df.copy()
        copied_df[gene_cols] = copied_df[gene_cols].notna().astype(int)
        copied_df['modality_score'] = copied_df[gene_cols].mean(axis=1)
        def agreement(row):
            available = row.replace(0, np.nan).dropna()
            if len(available) >= 2:
                return 1.0 / (1.0 + np.var(available.values))
            elif len(available) == 1:
                return 1.0
            return np.nan

        data_df_working = data_df.copy()
        data_df_working['agreement'] = raw_features_df.apply(agreement, axis=1)

        copied_df = copied_df[['ModelID', 'modality_score']]
        data_df_working = data_df_working[['ModelID', 'agreement']]
        data_df_working = data_df_working.merge(copied_df, on='ModelID', how='left')
        data_df_working['confidence_score'] = (data_df_working['modality_score'] + data_df_working['agreement']) / 2

        log.info(f"confidence_score: nulls={data_df_working['confidence_score'].isna().sum()} " f"of {len(data_df_working)}")
        return data_df_working
    except Exception as e:
        log.error(f"Error in calculateconfidencescore: {e}")
        raise


def calculatesimilarityscore(data_df, type):
    try:
        lambdavalue = 1.2
        log.info(f"calculatesimilarityscore (type={type}): input shape {data_df.shape}")
        if 'ModelID' not in data_df.columns:
            raise KeyError("data_df must contain a 'ModelID' column")
        gene_cols = [c for c in data_df.columns if c != 'ModelID']
        log.info(f"calculatesimilarityscore: {len(gene_cols)} cosine column(s)")
        data_df[gene_cols] = data_df[gene_cols].fillna(0)

        if type == "target":
            data_df['target_similarity'] = data_df[gene_cols].mean(axis=1)
            log.info(f"target_similarity: min={data_df['target_similarity'].min():.4f} "
                     f"max={data_df['target_similarity'].max():.4f}")
            return data_df[['ModelID', 'target_similarity']]
        elif type == "exclusion":
            data_df['exclusion_similarity'] = data_df[gene_cols].mean(axis=1)
            data_df['exclusion_similarity_percentile'] = data_df['exclusion_similarity'].rank(pct=True)
            data_df['exclusion_similarity_penalty'] = np.exp(-lambdavalue * data_df['exclusion_similarity_percentile'])
            log.info(f"exclusion_similarity_penalty: min={data_df['exclusion_similarity_penalty'].min():.4f} "
                     f"max={data_df['exclusion_similarity_penalty'].max():.4f}")
            return data_df[['ModelID', 'exclusion_similarity', 'exclusion_similarity_percentile', 'exclusion_similarity_penalty']]
        else:
            raise ValueError(f"type must be 'target' or 'exclusion', got {type!r}")
    except Exception as e:
        log.error(f"Error in calculatesimilarityscore (type={type}): {e}")
        raise


def calculatenetevidencescore(data_df, type):
    try:
        log.info(f"calculatenetevidencescore (type={type}): input shape {data_df.shape}")
        if type == "target":
            data_df['net_evidence'] = data_df['target_evidence']
            return data_df
        elif type == "exclusion":
            data_df['net_evidence'] = data_df['target_evidence'] * data_df['exclusion_penalty']
            return data_df
        else:
            raise ValueError(f"type must be 'target' or 'exclusion', got {type!r}")
    except Exception as e:
        log.error(f"Error in calculatenetevidencescore (type={type}): {e}")
        raise


def calculatenetsimilarityscore(data_df, type):
    try:
        log.info(f"calculatenetsimilarityscore (type={type}): input shape {data_df.shape}")
        if type == "target":
            data_df['net_similarity'] = data_df['target_similarity']
            return data_df
        elif type == "exclusion":
            data_df['net_similarity'] = data_df['target_similarity'] * data_df['exclusion_similarity_penalty']
            return data_df
        else:
            raise ValueError(f"type must be 'target' or 'exclusion', got {type!r}")
    except Exception as e:
        log.error(f"Error in calculatenetsimilarityscore (type={type}): {e}")
        raise


def finalscoring(data_df):
    try:
        log.info(f"finalscoring: input shape {data_df.shape}")
        for col in ['net_similarity', 'net_evidence', 'confidence_score']:
            if col not in data_df.columns:
                raise KeyError(f"finalscoring requires column '{col}' which is missing")
        data_df['final_score'] = (
            data_df['net_similarity'] * data_df['net_evidence'] * data_df['confidence_score'])
        log.info(f"final_score: nulls={data_df['final_score'].isna().sum()} of {len(data_df)}")
        return data_df
    except Exception as e:
        log.error(f"Error in finalscoring: {e}")
        raise