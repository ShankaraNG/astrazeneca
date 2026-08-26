import os
import numpy as np
import pandas as pd
import ml_build.utils as ut
from app.logger import get_logger

log = get_logger('AppPreprocessing')


def genetoesng(genelist):
    try:
        log.info(f"genetoesng: resolving {len(genelist) if genelist else 0} gene symbol(s) to ENSG")
        if not genelist:
            raise ValueError("genetoesng received an empty gene list")
        esngconverterdata_df = ut.cleaned_data_reader('lookup', 'gene_maps', 'gene_ensg_map.csv')
        if esngconverterdata_df is None or esngconverterdata_df.empty:
            raise ValueError("Mapping file 'gene_ensg_map.csv' is missing or empty.")
        matched_data_df = esngconverterdata_df[esngconverterdata_df['gene_symbol'].isin(genelist)]
        ensg_id_list = matched_data_df['ensg_id'].dropna().unique().tolist()
        found_symbols = set(matched_data_df['gene_symbol'].unique())
        missing_symbols = [g for g in genelist if g not in found_symbols]
        if missing_symbols:
            log.warning(f"genetoesng: {len(missing_symbols)} symbol(s) not found in map: {missing_symbols}")
        log.info(f"genetoesng: resolved {len(ensg_id_list)} ENSG id(s)")
        return ensg_id_list
    except Exception as e:
        log.error(f"Error converting genes to ENSG IDs: {e}")
        raise


def esngtoprotein(ensg_id_list):
    try:
        log.info(f"esngtoprotein: resolving proteins for {len(ensg_id_list)} ENSG id(s)")
        esngconverterproteindata_df = ut.cleaned_data_reader('lookup', 'gene_maps', 'gene_protein_map.csv')
        if esngconverterproteindata_df is None or esngconverterproteindata_df.empty:
            raise ValueError("Mapping file 'gene_protein_map.csv' is missing or empty.")
        input_set = set(ensg_id_list)
        matched_data_df = esngconverterproteindata_df[esngconverterproteindata_df['ensg_id'].isin(input_set)]
        found_proteins = matched_data_df['protein_id'].dropna().unique().tolist()
        null_protein_ensg = matched_data_df[matched_data_df['protein_id'].isna()]['ensg_id'].unique()
        existing_ensg_in_file = set(matched_data_df['ensg_id'].unique())
        completely_missing_ensg = input_set - existing_ensg_in_file
        unmatched_ensg_ids = set(null_protein_ensg).union(completely_missing_ensg)
        not_found_proteins = [f"prot_{ensg}" for ensg in unmatched_ensg_ids if ensg]
        log.info(f"esngtoprotein: {len(found_proteins)} protein(s) found, "
                 f"{len(not_found_proteins)} ENSG id(s) without a protein (0-fill placeholders)")
        return found_proteins, not_found_proteins
    except Exception as e:
        log.error(f"Error converting ENSG IDs to Protein IDs: {e}")
        raise


def extract_with_fallback(data_df, id_list, id_kind, prefix, filename):
    existing_columns = set(data_df.columns)
    id_list = list(dict.fromkeys(id_list))  # de-dupe, preserve order
    found = [i for i in id_list if i in existing_columns]
    missing = [i for i in id_list if i not in existing_columns]
    result = data_df[['ModelID'] + found].copy()
    for i in missing:
        result[f"{prefix}{i}"] = np.nan  # NaN: absent id -> modality counts as not-found
    log.info(f"{filename}: {id_kind} -> {len(found)} found, {len(missing)} NaN-filled (not found)")
    return result


def depmapreader(ensg_id_list):
    try:
        log.info(f"depmapreader: extracting {len(ensg_id_list)} ENSG column(s)")
        depmap_df = ut.cleaned_data_reader('scoring', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        if depmap_df is None or depmap_df.empty:
            raise ValueError("File '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv' is missing or empty.")
        out = extract_with_fallback(depmap_df, ensg_id_list, 'RNA/ENSG', 'DepMap_', 'depmap')
        log.info(f"depmapreader: returning shape {out.shape}")
        return out
    except Exception as e:
        log.error(f"Error finding the ENSG ID in the depmap table: {e}")
        raise


def hparnareader(ensg_id_list):
    try:
        log.info(f"hparnareader: extracting {len(ensg_id_list)} ENSG column(s)")
        hpa_rna_df = ut.cleaned_data_reader('scoring', 'gene expression', '1_4_hpa_rna_celline.csv')
        if hpa_rna_df is None or hpa_rna_df.empty:
            raise ValueError("File '1_4_hpa_rna_celline.csv' is missing or empty.")
        drop_extra = [c for c in ['Cell line', 'Cellosaurus ID'] if c in hpa_rna_df.columns]
        if drop_extra:
            hpa_rna_df = hpa_rna_df.drop(columns=drop_extra)
        out = extract_with_fallback(hpa_rna_df, ensg_id_list, 'HPA/ENSG', 'HPA_', 'hpa')
        log.info(f"hparnareader: returning shape {out.shape}")
        return out
    except Exception as e:
        log.error(f"Error finding ENSG ID in the HPA RNA table: {e}")
        raise


def proteinreader(protein_id_list):
    try:
        log.info(f"proteinreader: extracting {len(protein_id_list)} protein column(s)")
        protein_df = ut.cleaned_data_reader('scoring', 'gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted.csv')
        if protein_df is None or protein_df.empty:
            raise ValueError("File '4_Harmonized_MS_CCLE_Gygi_subsetted.csv' is missing or empty.")
        out = extract_with_fallback(protein_df, protein_id_list, 'protein', 'Prot_', 'protein')
        log.info(f"proteinreader: returning shape {out.shape}")
        return out
    except Exception as e:
        log.error(f"Error finding protein ID in the proteomics table: {e}")
        raise


def similarityreader(ensg_id_list):
    try:
        log.info(f"similarityreader: extracting {len(ensg_id_list)} gene cosine column(s)")
        cosinematrix = ut.cleaned_data_reader('scoring', 'cosine_matrix', 'cosinematrix.csv')
        if cosinematrix is None or cosinematrix.empty:
            raise ValueError("File 'cosinematrix.csv' is missing or empty.")
        out = extract_with_fallback(cosinematrix, ensg_id_list, 'cosine/ENSG', 'similarity_', 'cosine')
        log.info(f"similarityreader: returning shape {out.shape}")
        return out
    except Exception as e:
        log.error(f"Error finding ENSG ID in the Similarity table: {e}")
        raise


def diseasecheker(disease):
    try:
        log.info(f"diseasecheker: resolving ModelIDs for disease '{disease}'")
        disease_df = ut.cleaned_data_reader('lookup', 'master', 'master_lookup.csv')
        if disease_df is None or disease_df.empty:
            raise ValueError("Mapping file 'master_lookup.csv' is missing or empty.")
        disease_df = disease_df[disease_df['primary_disease'] == disease]
        requiredlistofmasterid = set(disease_df['ModelID'].dropna())
        if not requiredlistofmasterid:
            raise ValueError(f"No cell lines found for disease '{disease}'")
        log.info(f"diseasecheker: {len(requiredlistofmasterid)} ModelID(s) match disease '{disease}'")
        return requiredlistofmasterid
    except Exception as e:
        log.error(f"Error from the disease checker: {e}")
        raise


def datacombiner(data_df):
    try:
        log.info(f"datacombiner: enriching {len(data_df)} row(s) with master_lookup identity")
        masterlookup_df = ut.cleaned_data_reader('lookup', 'master', 'master_lookup.csv')
        if masterlookup_df is None or masterlookup_df.empty:
            raise ValueError("Mapping file 'master_lookup.csv' is missing or empty.")
        required_columns = ['ModelID', 'CVCL_ID', 'CCLE_Name', 'cell_line_name',
                            'stripped_cell_line_name', 'primary_disease', 'lineage']
        keep = [c for c in required_columns if c in masterlookup_df.columns]
        masterlookup_df = masterlookup_df[keep]
        data_df = data_df.merge(masterlookup_df, on='ModelID', how='left')
        priority_columns = ['cell_line_name', 'stripped_cell_line_name', 'ModelID', 'CVCL_ID',
                            'CCLE_Name', 'primary_disease', 'lineage']
        priority_columns = [c for c in priority_columns if c in data_df.columns]
        other_cols = [c for c in data_df.columns if c not in priority_columns]
        data_df = data_df[priority_columns + other_cols]
        log.info(f"datacombiner: returning shape {data_df.shape}")
        return data_df
    except Exception as e:
        log.error(f"Error combining master data into results: {e}")
        raise


def alternativedatacombiner(data_df):
    try:
        log.info(f"alternativedatacombiner: enriching {len(data_df)} alternative row(s)")
        masterlookup_df = ut.cleaned_data_reader('lookup', 'master', 'master_lookup.csv')
        if masterlookup_df is None or masterlookup_df.empty:
            raise ValueError("Mapping file 'master_lookup.csv' is missing or empty.")
        required_columns = ['ModelID', 'CVCL_ID', 'CCLE_Name', 'cell_line_name',
                            'stripped_cell_line_name', 'primary_disease', 'lineage']
        keep = [c for c in required_columns if c in masterlookup_df.columns]
        masterlookup_df = masterlookup_df[keep].rename(columns={'ModelID': 'AlternativeModelID'})
        data_df = data_df.merge(masterlookup_df, on='AlternativeModelID', how='left')
        priority_columns = ['cell_line_name', 'stripped_cell_line_name', 'AlternativeModelID',
                            'ModelID', 'CVCL_ID', 'CCLE_Name', 'primary_disease', 'lineage']
        priority_columns = [c for c in priority_columns if c in data_df.columns]
        other_cols = [c for c in data_df.columns if c not in priority_columns]
        data_df = data_df[priority_columns + other_cols]
        log.info(f"alternativedatacombiner: returning shape {data_df.shape}")
        return data_df
    except Exception as e:
        log.error(f"Error combining master data into alternatives: {e}")
        raise