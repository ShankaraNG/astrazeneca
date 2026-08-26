import os
import numpy as np
import pandas as pd
import ml_build.utils as ut
from app.logger import get_logger

log = get_logger('Mutation')


# def mutationflager(data_df, geneslist):
#     try:
#         log.info(f"mutationflager: {len(data_df)} cell line(s), {len(geneslist)} gene(s)")
#         copied_df = data_df.copy()
#         omnicsmutation_df = ut.cleaned_data_reader('scoring', 'gene properties', '6_OmicsSomaticMutationsProfile.csv')
#         if omnicsmutation_df is None or omnicsmutation_df.empty:
#             raise ValueError("6_OmicsSomaticMutationsProfile.csv is empty or missing.")
#         log.info(f"mutationflager: mutation table shape {omnicsmutation_df.shape}")

#         omnicsmutation_df = omnicsmutation_df.dropna(subset=['ModelID'])
#         omnicsmutation_df = omnicsmutation_df[omnicsmutation_df['EnsemblGeneID'].isin(geneslist)]
#         modelidlistwithmutation = set(omnicsmutation_df['ModelID'])
#         log.info(f"mutationflager: {len(modelidlistwithmutation)} distinct ModelID(s) with a matching mutation")

#         copied_df['mutation_flag'] = np.where(
#             copied_df['ModelID'].isin(modelidlistwithmutation), 'Yes', 'No')
#         log.info(f"mutationflager: {(copied_df['mutation_flag'] == 'Yes').sum()} cell line(s) flagged Yes")
#         return copied_df
#     except Exception as e:
#         log.error(f"Error executing mutationflager: {e}")
#         raise

def mutationflager(data_df, targetgeneslist, exclusiongeneslist=None):
    try:
        exclusiongeneslist = exclusiongeneslist or []
        target_set = set(targetgeneslist)
        exclusion_set = set(exclusiongeneslist)
        combined_set = target_set | exclusion_set
        log.info(f"mutationflager: {len(data_df)} cell line(s), "
                 f"{len(target_set)} target gene(s), {len(exclusion_set)} exclusion gene(s)")
        copied_df = data_df.copy()
        omnicsmutation_df = ut.cleaned_data_reader('scoring', 'gene properties', '6_OmicsSomaticMutationsProfile.csv')
        if omnicsmutation_df is None or omnicsmutation_df.empty:
            raise ValueError("6_OmicsSomaticMutationsProfile.csv is empty or missing.")
        log.info(f"mutationflager: mutation table shape {omnicsmutation_df.shape}")
        omnicsmutation_df = omnicsmutation_df.dropna(subset=['ModelID'])
        omnicsmutation_df = omnicsmutation_df[omnicsmutation_df['EnsemblGeneID'].isin(combined_set)]
        exclusion_model_ids = set(
            omnicsmutation_df.loc[omnicsmutation_df['EnsemblGeneID'].isin(exclusion_set), 'ModelID'].unique())
        target_model_ids = set(
            omnicsmutation_df.loc[omnicsmutation_df['EnsemblGeneID'].isin(target_set), 'ModelID'].unique())
        all_mutation_model_ids = exclusion_model_ids | target_model_ids
        log.info(f"mutationflager: {len(all_mutation_model_ids)} distinct ModelID(s) with a matching mutation; "
                 f"{len(exclusion_model_ids)} exclusion-involved, {len(target_model_ids)} target-involved")
        copied_df['mutation_flag'] = np.where(
            copied_df['ModelID'].isin(all_mutation_model_ids), 'Yes', 'No')
        copied_df['mutation_type'] = np.select(
            [copied_df['ModelID'].isin(exclusion_model_ids),
             copied_df['ModelID'].isin(target_model_ids)],
            ['exclusion', 'target'],
            default='No')
        log.info(f"mutationflager: mutation_type counts -> "
                 f"{copied_df['mutation_type'].value_counts().to_dict()}")
        return copied_df
    except Exception as e:
        log.error(f"Error executing mutationflager: {e}")
        raise

def remove_mutation_models(data_df):
    try:
        log.info(f"remove_mutation_models: {len(data_df)} row(s) in")
        if 'mutation_flag' not in data_df.columns:
            raise KeyError("Input must be processed by mutationflager() first (no 'mutation_flag' column).")
        out = data_df[data_df['mutation_flag'] == 'No']
        log.info(f"remove_mutation_models: {len(out)} row(s) retained after dropping mutation-positive")
        return out
    except Exception as e:
        log.error(f"Error executing remove_mutation_models: {e}")
        raise

def require_mutation_models(data_df):
    try:
        log.info(f"require_mutation_models: {len(data_df)} row(s) in")
        if 'mutation_flag' not in data_df.columns or 'mutation_type' not in data_df.columns:
            raise KeyError("Input must be processed by mutationflager() first (needs 'mutation_flag' and 'mutation_type').")
        out = data_df[(data_df['mutation_flag'] == 'Yes') & (data_df['mutation_type'] != 'exclusion')]
        log.info(f"require_mutation_models: {len(out)} row(s) retained after dropping mutation-positive")
        return out
    except Exception as e:
        log.error(f"Error executing require_mutation_models: {e}")
        raise


def getthemutationreferencetable(data_df, geneslist):
    try:
        log.info(f"getthemutationreferencetable: {len(data_df)} cell line(s), {len(geneslist)} gene(s)")
        copied_df = data_df.copy()
        omnicsmutation_df = ut.cleaned_data_reader('scoring', 'gene properties', '6_OmicsSomaticMutationsProfile.csv')
        if omnicsmutation_df is None or omnicsmutation_df.empty:
            raise ValueError("6_OmicsSomaticMutationsProfile.csv is empty or missing.")

        omnicsmutation_df = omnicsmutation_df.dropna(subset=['ModelID'])
        omnicsmutation_df = omnicsmutation_df[omnicsmutation_df['EnsemblGeneID'].isin(geneslist)]

        if 'mutation_flag' in copied_df.columns:
            copied_df = copied_df[copied_df['mutation_flag'] == 'Yes']
        modelidlistwithmutation = set(copied_df['ModelID'])
        omnicsmutation_df = omnicsmutation_df[omnicsmutation_df['ModelID'].isin(modelidlistwithmutation)]

        requested_columns = ['ModelID', 'EnsemblGeneID', 'ProteinChange', 'DNAChange', 'VariantType',
                             'VariantInfo', 'GT', 'AMPathogenicity', 'AMClass', 'VepImpact', 'Hotspot',
                             'ProveanPrediction', 'LikelyLoF', 'OncogeneHighImpact', 'TumorSuppressorHighImpact']
        present = [c for c in requested_columns if c in omnicsmutation_df.columns]
        missing = [c for c in requested_columns if c not in omnicsmutation_df.columns]
        if missing:
            log.warning(f"getthemutationreferencetable: columns not in mutation file: {missing}")
        omnicsmutation_df = omnicsmutation_df[present]
        log.info(f"getthemutationreferencetable: returning {len(omnicsmutation_df)} reference row(s)")
        return omnicsmutation_df
    except Exception as e:
        log.error(f"Error executing getthemutationreferencetable: {e}")
        raise