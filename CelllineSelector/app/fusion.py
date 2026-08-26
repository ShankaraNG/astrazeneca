import os
import gc
import numpy as np
import pandas as pd
import ml_build.utils as ut
from app.logger import get_logger


log = get_logger('Fusion')


# def fusionflagger(data_df, geneslist):
#     try:
#         log.info(f"fusionflagger: {len(data_df)} cell line(s), {len(geneslist)} gene(s)")
#         copied_df = data_df.copy()
#         omnicsfusion_df = ut.cleaned_data_reader('scoring', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
#         if omnicsfusion_df is None or omnicsfusion_df.empty:
#             raise ValueError("5_OmicsFusionFilteredSupplementary.csv is empty or missing.")
#         log.info(f"fusionflagger: fusion table shape {omnicsfusion_df.shape}")
#         filtergene_df = omnicsfusion_df[
#             omnicsfusion_df['gene1_ENSG_ID'].isin(geneslist) &
#             omnicsfusion_df['gene2_ENSG_ID'].isin(geneslist)
#         ].copy()
#         fusion_model_ids = filtergene_df['ModelID'].unique()
#         log.info(f"fusionflagger: {len(filtergene_df)} matching fusion row(s) and {len(fusion_model_ids)} distinct flagged ModelID(s)")
#         log.info("fusionflagger: releasing fusion table frames from memory")
#         del omnicsfusion_df, filtergene_df
#         gc.collect()
#         copied_df['fusion_flag'] = np.where(copied_df['ModelID'].isin(fusion_model_ids), 'Yes', 'No')
#         log.info(f"fusionflagger: {(copied_df['fusion_flag'] == 'Yes').sum()} cell line(s) flagged Yes")
#         return copied_df
#     except Exception as e:
#         log.error(f"Error executing fusionflagger: {e}")
#         raise

# def fusionflagger(data_df, targetgeneslist, exclusiongeneslist=None):
#     try:
#         exclusiongeneslist = exclusiongeneslist or []
#         target_set = set(targetgeneslist)
#         exclusion_set = set(exclusiongeneslist)
#         combined_set = target_set | exclusion_set
#         log.info(f"fusionflagger: {len(data_df)} cell line(s), "
#                  f"{len(target_set)} target gene(s), {len(exclusion_set)} exclusion gene(s)")
#         copied_df = data_df.copy()
#         omnicsfusion_df = ut.cleaned_data_reader('scoring', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
#         if omnicsfusion_df is None or omnicsfusion_df.empty:
#             raise ValueError("5_OmicsFusionFilteredSupplementary.csv is empty or missing.")
#         log.info(f"fusionflagger: fusion table shape {omnicsfusion_df.shape}")
#         # omnicsfusion_df = omnicsfusion_df[omnicsfusion_df['ModelConditionID'] == 'Yes']
#         filtergene_df = omnicsfusion_df[
#             omnicsfusion_df['gene1_ENSG_ID'].isin(combined_set) &
#             omnicsfusion_df['gene2_ENSG_ID'].isin(combined_set)
#         ].copy()
#         involves_exclusion = (
#             filtergene_df['gene1_ENSG_ID'].isin(exclusion_set) |
#             filtergene_df['gene2_ENSG_ID'].isin(exclusion_set)
#         )
#         exclusion_model_ids = set(filtergene_df.loc[involves_exclusion, 'ModelID'].unique())
#         target_model_ids = set(filtergene_df.loc[~involves_exclusion, 'ModelID'].unique())
#         all_fusion_model_ids = exclusion_model_ids | target_model_ids
#         log.info(f"fusionflagger: {len(filtergene_df)} matching fusion row(s); "
#                  f"{len(exclusion_model_ids)} exclusion-involved, "
#                  f"{len(target_model_ids)} target-only ModelID(s)")
#         log.info("fusionflagger: releasing fusion table frames from memory")
#         del omnicsfusion_df, filtergene_df
#         gc.collect()
#         copied_df['fusion_flag'] = np.where(
#             copied_df['ModelID'].isin(all_fusion_model_ids), 'Yes', 'No')
#         copied_df['fusion_type'] = np.select(
#             [copied_df['ModelID'].isin(exclusion_model_ids),
#              copied_df['ModelID'].isin(target_model_ids)],
#             ['exclusion', 'target'],
#             default='No')
#         log.info(f"fusionflagger: fusion_type counts -> "
#                  f"{copied_df['fusion_type'].value_counts().to_dict()}")
#         return copied_df
#     except Exception as e:
#         log.error(f"Error executing fusionflagger: {e}")
#         raise

def fusionflagger(data_df, targetgeneslist, exclusiongeneslist=None):
    try:
        exclusiongeneslist = exclusiongeneslist or []
        target_set = set(targetgeneslist)
        exclusion_set = set(exclusiongeneslist)
        combined_set = target_set | exclusion_set
        log.info(f"fusionflagger: {len(data_df)} cell line(s), "
                 f"{len(target_set)} target gene(s), {len(exclusion_set)} exclusion gene(s)")
        copied_df = data_df.copy()
        omnicsfusion_df = ut.cleaned_data_reader('scoring', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
        if omnicsfusion_df is None or omnicsfusion_df.empty:
            raise ValueError("5_OmicsFusionFilteredSupplementary.csv is empty or missing.")
        log.info(f"fusionflagger: fusion table shape {omnicsfusion_df.shape}")
        single_target = (len(target_set) == 1)

        if single_target:
            log.info("fusionflagger: single target gene, no exclusion -> OR match (gene as either partner)")
            filtergene_df = omnicsfusion_df[
                omnicsfusion_df['gene1_ENSG_ID'].isin(combined_set) |
                omnicsfusion_df['gene2_ENSG_ID'].isin(combined_set)
            ].copy()
            exclusion_model_ids = set()
            target_model_ids = set(filtergene_df['ModelID'].unique())
        else:
            filtergene_df = omnicsfusion_df[
                omnicsfusion_df['gene1_ENSG_ID'].isin(combined_set) &
                omnicsfusion_df['gene2_ENSG_ID'].isin(combined_set)
            ].copy()
            involves_exclusion = (
                filtergene_df['gene1_ENSG_ID'].isin(exclusion_set) |
                filtergene_df['gene2_ENSG_ID'].isin(exclusion_set)
            )
            exclusion_model_ids = set(filtergene_df.loc[involves_exclusion, 'ModelID'].unique())
            target_model_ids = set(filtergene_df.loc[~involves_exclusion, 'ModelID'].unique())

        all_fusion_model_ids = exclusion_model_ids | target_model_ids
        log.info(f"fusionflagger: {len(filtergene_df)} matching fusion row(s); "
                 f"{len(exclusion_model_ids)} exclusion-involved, "
                 f"{len(target_model_ids)} target-only ModelID(s)")
        log.info("fusionflagger: releasing fusion table frames from memory")
        del omnicsfusion_df, filtergene_df
        gc.collect()
        copied_df['fusion_flag'] = np.where(
            copied_df['ModelID'].isin(all_fusion_model_ids), 'Yes', 'No')
        copied_df['fusion_type'] = np.select(
            [copied_df['ModelID'].isin(exclusion_model_ids),
             copied_df['ModelID'].isin(target_model_ids)],
            ['exclusion', 'target'],
            default='No')
        log.info(f"fusionflagger: fusion_type counts -> "
                 f"{copied_df['fusion_type'].value_counts().to_dict()}")
        return copied_df
    except Exception as e:
        log.error(f"Error executing fusionflagger: {e}")
        raise

def remove_fusion_models(flagged_df):
    try:
        log.info(f"remove_fusion_models: {len(flagged_df)} row(s) in")
        if 'fusion_flag' not in flagged_df.columns:
            raise KeyError("Input must be processed by fusionflagger() first (no 'fusion_flag' column).")
        out = flagged_df[flagged_df['fusion_flag'] == 'No']
        log.info(f"remove_fusion_models: {len(out)} row(s) retained after dropping fusion-positive")
        return out
    except Exception as e:
        log.error(f"Error executing remove_fusion_models: {e}")
        raise

def require_fusion_models(flagged_df):
    try:
        log.info(f"require_fusion_models: {len(flagged_df)} row(s) in")
        if 'fusion_flag' not in flagged_df.columns or 'fusion_type' not in flagged_df.columns:
            raise KeyError("Input must be processed by fusionflagger() first (needs 'fusion_flag' and 'fusion_type').")
        out = flagged_df[(flagged_df['fusion_flag'] == 'Yes') & (flagged_df['fusion_type'] != 'exclusion')]
        # out = flagged_df
        log.info(f"require_fusion_models: {len(out)} row(s) retained after dropping fusion-positive")
        return out
    except Exception as e:
        log.error(f"Error executing require_fusion_models: {e}")
        raise


# def getthefusionreferencetable(top_df, geneslist):
#     try:
#         log.info(f"getthefusionreferencetable: {len(top_df)} top cell line(s), {len(geneslist)} gene(s)")
#         copied_df = top_df.copy()
#         omnicsfusion_df = ut.cleaned_data_reader('scoring', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
#         if omnicsfusion_df is None or omnicsfusion_df.empty:
#             raise ValueError("5_OmicsFusionFilteredSupplementary.csv is empty or missing.")
#         filtergene_df = omnicsfusion_df[
#             omnicsfusion_df['gene1_ENSG_ID'].isin(geneslist) &
#             omnicsfusion_df['gene2_ENSG_ID'].isin(geneslist)
#         ].copy()
#         log.info("getthefusionreferencetable: releasing full fusion table from memory")
#         del omnicsfusion_df
#         gc.collect()
#         if 'fusion_flag' in copied_df.columns:
#             copied_df = copied_df[copied_df['fusion_flag'] == 'Yes']
#         modelidlist = set(copied_df['ModelID'])
#         filtergene_df = filtergene_df[filtergene_df['ModelID'].isin(modelidlist)]
#         requested_columns = [
#             'ModelID', 'ModelConditionID', 'CanonicalFusionName', 'TotalReadsInSample',
#             'TotalReadsSupportingFusion', 'FFPM', 'confidence', 'gene1_ENSG_ID', 'gene2_ENSG_ID'
#         ]
#         present = [c for c in requested_columns if c in filtergene_df.columns]
#         missing = [c for c in requested_columns if c not in filtergene_df.columns]
#         if missing:
#             log.warning(f"getthefusionreferencetable: columns not in fusion file: {missing}")
#         filtergene_df = filtergene_df[present]
#         log.info(f"getthefusionreferencetable: returning {len(filtergene_df)} reference row(s)")
#         return filtergene_df
#     except Exception as e:
#         log.error(f"Error executing getthefusionreferencetable: {e}")
#         raise

def getthefusionreferencetable(top_df, targetgeneslist, exclusiongeneslist=None):
    try:
        exclusiongeneslist = exclusiongeneslist or []
        target_set = set(targetgeneslist)
        exclusion_set = set(exclusiongeneslist)
        combined_set = target_set | exclusion_set
        log.info(f"getthefusionreferencetable: {len(top_df)} top cell line(s), "
                 f"{len(target_set)} target gene(s), {len(exclusion_set)} exclusion gene(s)")
        copied_df = top_df.copy()
        omnicsfusion_df = ut.cleaned_data_reader('scoring', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
        if omnicsfusion_df is None or omnicsfusion_df.empty:
            raise ValueError("5_OmicsFusionFilteredSupplementary.csv is empty or missing.")
        
        single_target = (len(target_set) == 1)
        if single_target:
            filtergene_df = omnicsfusion_df[
                omnicsfusion_df['gene1_ENSG_ID'].isin(combined_set) |
                omnicsfusion_df['gene2_ENSG_ID'].isin(combined_set)
            ].copy()
        else:
            filtergene_df = omnicsfusion_df[
                omnicsfusion_df['gene1_ENSG_ID'].isin(combined_set) &
                omnicsfusion_df['gene2_ENSG_ID'].isin(combined_set)
            ].copy()

        log.info("getthefusionreferencetable: releasing full fusion table from memory")
        del omnicsfusion_df
        gc.collect()
        if 'fusion_flag' in copied_df.columns:
            copied_df = copied_df[copied_df['fusion_flag'] == 'Yes']
        modelidlist = set(copied_df['ModelID'])
        filtergene_df = filtergene_df[filtergene_df['ModelID'].isin(modelidlist)]
        requested_columns = [
            'ModelID', 'ModelConditionID', 'CanonicalFusionName', 'TotalReadsInSample',
            'TotalReadsSupportingFusion', 'FFPM', 'confidence', 'gene1_ENSG_ID', 'gene2_ENSG_ID'
        ]
        present = [c for c in requested_columns if c in filtergene_df.columns]
        missing = [c for c in requested_columns if c not in filtergene_df.columns]
        if missing:
            log.warning(f"getthefusionreferencetable: columns not in fusion file: {missing}")
        filtergene_df = filtergene_df[present]
        log.info(f"getthefusionreferencetable: returning {len(filtergene_df)} reference row(s)")
        return filtergene_df
    except Exception as e:
        log.error(f"Error executing getthefusionreferencetable: {e}")
        raise