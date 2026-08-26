import pandas as pd
import numpy as np

def calculateevidencescore(data_df, type):
    try:
        gene_cols = data_df.columns[1:]
        data_df[gene_cols] = data_df[gene_cols].fillna(0)
        lambda_val = 1.5
        if type == "target":
            data_df['target_evidence'] = data_df[gene_cols].mean(axis=1)
            return data_df[['ModelID', 'target_evidence']]
        elif type == "exclusion":
            data_df['exclusion_evidence'] = data_df[gene_cols].mean(axis=1)
            data_df['exclusion_percentile'] = data_df['exclusion_evidence'].rank(pct=True)
            data_df['exclusion_penalty'] = np.exp(-lambda_val * data_df['exclusion_percentile'])
            return data_df[['ModelID', 'exclusion_evidence', 'exclusion_percentile', 'exclusion_penalty']]
        else:
            raise ValueError()
    except Exception as e:
        print(e)
        raise


def calculateconfidencescore(data_df):
    try:
        # Save a completely raw copy of the gene data before any mutations happen
        gene_cols = data_df.columns[1:]
        raw_features_df = data_df[gene_cols].copy()

        copied_df = data_df.copy()
        
        # 2. Apply your explicit truth layout rule:
        #    - NaN becomes 0
        #    - 0 becomes 1
        #    - Any other numeric value becomes 1
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
        return data_df_working
    except Exception as e:
        print(e)
        raise


def calculatesimilarityscore(data_df, type):
    try:
        gene_cols = data_df.columns[1:]
        data_df[gene_cols] = data_df[gene_cols].fillna(0)
        lambda_val = 1.5
        if type == "target":
            data_df['target_similarity'] = data_df[gene_cols].mean(axis=1)
            return data_df[['ModelID', 'target_similarity']]
        elif type == "exclusion":
            data_df['exclusion_similarity'] = data_df[gene_cols].mean(axis=1)
            data_df['exclusion_similarity_percentile'] = data_df['exclusion_similarity'].rank(pct=True)
            data_df['exclusion_similarity_penalty'] = np.exp(-lambda_val * data_df['exclusion_similarity_percentile'])
            return data_df[['ModelID', 'exclusion_similarity', 'exclusion_similarity_percentile', 'exclusion_similarity_penalty']]
        else:
            raise ValueError()
    except Exception as e:
        print(e)
        raise

def calculatenetevidencescore(data_df, type):
    try:
        if type == "target":
            data_df['net_evidence'] = data_df['target_evidence']
            return data_df
        elif type == "exclusion":
            data_df['net_evidence'] = data_df['target_evidence']*data_df['exclusion_penalty']
            return data_df
        else:
            raise ValueError()
    except Exception as e:
        print(e)
        raise

def calculatenetsimilarityscore(data_df, type):
    try:
        if type == "target":
            data_df['net_similarity'] = data_df['target_similarity']
            return data_df
        elif type == "exclusion":
            data_df['net_similarity'] = data_df['target_similarity']*data_df['exclusion_similarity_penalty']
            return data_df
        else:
            raise ValueError()
    except Exception as e:
        print(e)
        raise

def finalscoring(data_df):
    try:
        data_df['final_score'] = data_df['net_similarity']*data_df['net_evidence']*data_df['confidence_score']
        return data_df
    except Exception as e:
        print(e)
        raise