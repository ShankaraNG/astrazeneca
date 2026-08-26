import os
import numpy as np
import pandas as pd
import ml_build.utils as ut

def genetoesng(genelist):
    try:
        esngconverterdata_df = ut.cleaned_data_reader('lookup', 'gene_maps', 'gene_ensg_map.csv')
        if esngconverterdata_df is None or esngconverterdata_df.empty:
            raise ValueError("Mapping file 'gene_ensg_map.csv' is missing or empty.")
        matched_data_df = esngconverterdata_df[esngconverterdata_df['gene_symbol'].isin(genelist)]
        ensg_id_list = matched_data_df['ensg_id'].dropna().unique().tolist()
        return ensg_id_list
    except Exception as e:
        print(f"Error converting genes to ENSG IDs: {e}")
        raise


def esngtoprotein(ensg_id_list):
    try:
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
        return found_proteins, not_found_proteins
        
    except Exception as e:
        print(f"Error converting ENSG IDs to Protein IDs: {e}")
        raise


def depmapreader(ensg_id_list):
    try:
        depmap_df = ut.cleaned_data_reader('scoring', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        if depmap_df is None or depmap_df.empty:
            raise ValueError("Mapping file '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv' is missing or empty.")
        existing_columns = set(depmap_df.columns)
        ensg_id_list = set(ensg_id_list)
        found_ensg = [ensg for ensg in ensg_id_list if ensg in existing_columns]
        missing_ensg = [ensg for ensg in ensg_id_list if ensg not in existing_columns]
        columns_to_extract = ['ModelID'] + found_ensg
        depmap_df = depmap_df[columns_to_extract].copy()
        for ensg in missing_ensg:
            fallback_column_name = f"DepMap_{ensg}"
            depmap_df[fallback_column_name] = np.nan
        return depmap_df     
    except Exception as e:
        print(f"Error finding the ESNG ID in the depmap table: {e}")
        raise

def hparnareader(ensg_id_list):
    try:
        hpa_rna_df = ut.cleaned_data_reader('scoring', 'gene expression', '1_4_hpa_rna_celline.csv')
        if hpa_rna_df is None or hpa_rna_df.empty:
            raise ValueError("Mapping file '1_4_hpa_rna_celline.csv' is missing or empty.")
        existing_columns = set(hpa_rna_df.columns)
        found_ensg = [ensg for ensg in ensg_id_list if ensg in existing_columns]
        missing_ensg = [ensg for ensg in ensg_id_list if ensg not in existing_columns]
        columns_to_extract = ['ModelID'] + found_ensg
        hpa_rna_df = hpa_rna_df[columns_to_extract]
        for ensg in missing_ensg:
            fallback_column_name = f"HPA_{ensg}"
            hpa_rna_df[fallback_column_name] = np.nan
        return hpa_rna_df    
    except Exception as e:
        print(f"Error finding ENSG ID in the HPA RNA table: {e}")
        raise

def proteinreader(protein_id_list):
    try:
        protein_df = ut.cleaned_data_reader('scoring', 'gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted.csv')
        if protein_df is None or protein_df.empty:
            raise ValueError("Mapping file '4_Harmonized_MS_CCLE_Gygi_subsetted.csv' is missing or empty.")
        existing_columns = set(protein_df.columns)
        found_protein = [protein for protein in protein_id_list if protein in existing_columns]
        missing_protein = [protein for protein in protein_id_list if protein not in existing_columns]
        columns_to_extract = ['ModelID'] + found_protein
        protein_df = protein_df[columns_to_extract]
        for prot in missing_protein:
            fallback_column_name = f"Prot_{prot}"
            protein_df[fallback_column_name] = np.nan
        return protein_df    
    except Exception as e:
        print(f"Error finding ENSG ID in the HPA RNA table: {e}")
        raise

def similarityreader(ensg_id_list):
    try:
        cosinematrix = ut.cleaned_data_reader('scoring', 'cosine_matrix', 'cosinematrix.csv')
        if cosinematrix is None or cosinematrix.empty:
            raise ValueError("Mapping file 'cosinematrix.csv' is missing or empty.")
        existing_columns = set(cosinematrix.columns)
        found_geneid = [geneid for geneid in ensg_id_list if geneid in existing_columns]
        missing_geneid = [geneid for geneid in ensg_id_list if geneid not in existing_columns]
        columns_to_extract = ['ModelID'] + found_geneid
        cosinematrix_df = cosinematrix[columns_to_extract]
        for gene in missing_geneid:
            fallback_column_name = f"similarity_{gene}"
            missing_geneid[fallback_column_name] = np.nan
        return cosinematrix_df    
    except Exception as e:
        print(f"Error finding ENSG ID in the Similarity table: {e}")
        raise

def diseasecheker(disease):
    try:
        disease_df = ut.cleaned_data_reader('lookup', 'master', 'master_lookup.csv')
        if disease_df is None or disease_df.empty:
            raise ValueError("Mapping file 'master_lookup.csv' is missing or empty.")
        disease_df = disease_df[disease_df['primary_disease'] == disease]
        requiredlistofmasterid = set(disease_df['primary_disease'])
        if requiredlistofmasterid is None:
            raise Exception("Unable to get the required list")
        return requiredlistofmasterid
    except Exception as e:
        print(f"Error from the disease checker: {e}")
        raise

def datacombiner(data_df):
    try:
        masterlookup_df = ut.cleaned_data_reader('lookup', 'master', 'master_lookup.csv')
        if masterlookup_df is None or masterlookup_df.empty:
            raise ValueError("Mapping file 'master_lookup.csv' is missing or empty.")
        required_columns = ['ModelID', 'CVCL_ID','CCLE_Name','cell_line_name','stripped_cell_line_name','primary_disease','lineage']
        masterlookup_df = masterlookup_df[required_columns]
        data_df = data_df.merge(masterlookup_df, on='ModelID', how='left')
        priority_columns = required_columns = ['cell_line_name','stripped_cell_line_name', 'ModelID' ,'CVCL_ID','CCLE_Name','primary_disease','lineage']
        other_cols = [col for col in data_df.columns if col not in priority_columns]
        data_df = data_df[priority_columns + other_cols]
        return data_df
    except Exception as e:
        print(f"Error finding other data to add it up: {e}")
        raise

def alternativedatacombiner(data_df):
    try:
        masterlookup_df = ut.cleaned_data_reader('lookup', 'master', 'master_lookup.csv')
        if masterlookup_df is None or masterlookup_df.empty:
            raise ValueError("Mapping file 'master_lookup.csv' is missing or empty.")
        required_columns = ['ModelID', 'CVCL_ID','CCLE_Name','cell_line_name','stripped_cell_line_name','primary_disease','lineage']
        masterlookup_df = masterlookup_df[required_columns].rename(columns={'ModelID' : 'AlternativeModelID'})
        data_df = data_df.merge(masterlookup_df, on='AlternativeModelID', how='left')
        priority_columns = required_columns = ['cell_line_name','stripped_cell_line_name', 'AlternativeModelID', 'ModelID' ,'CVCL_ID','CCLE_Name','primary_disease','lineage']
        other_cols = [col for col in data_df.columns if col not in priority_columns]
        data_df = data_df[priority_columns + other_cols]
        return data_df
    except Exception as e:
        print(f"Error finding other data to add it up: {e}")
        raise






