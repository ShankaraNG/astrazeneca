import os
import sys
import pandas as pd


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def data_reader(subfolder, filename):
    try:
        data_sub_path = os.path.join(base_path, "data", "raw", subfolder)
        if not os.path.isdir(data_sub_path):
            raise Exception("Data path doesnt exists")
        if filename == "1_4_hpa_rna_celline":
            filepath = os.path.join(data_sub_path,"1_4_hpa_rna_celline.tsv")
            if not os.path.isfile(filepath):
                raise Exception("Data File 1_4_hpa_rna_celline.tsv doesnt exists")
            hpa_rna_df = pd.read_csv(filepath, sep='\t')
            return hpa_rna_df
        elif filename == "2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile":
            filepath = os.path.join(data_sub_path,"2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data File 2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv doesnt exists")
            depmap_omicsexpressionallgenes_df = pd.read_csv(filepath)
            return depmap_omicsexpressionallgenes_df
        elif filename == "3_GEOexpression":
            filepath = os.path.join(data_sub_path,"3_GEOexpression.txt")
            if not os.path.isfile(filepath):
                raise Exception("Data File 3_GEOexpression.txt doesnt exists")
            geoexpression_df = pd.read_csv(filepath, sep='\t')
            return geoexpression_df
        elif filename == "4_Harmonized_MS_CCLE_Gygi_subsetted":
            filepath = os.path.join(data_sub_path,"4_Harmonized_MS_CCLE_Gygi_subsetted.csv")
            if not os.path.isfile(filepath):
                raise Exception("4_Harmonized_MS_CCLE_Gygi_subsetted.csv doesnt exists")
            harmonized_ms_celline_df = pd.read_csv(filepath)
            return harmonized_ms_celline_df
        elif filename == "5_OmicsFusionFilteredSupplementary":
            filepath = os.path.join(data_sub_path,"5_OmicsFusionFilteredSupplementary.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 5_OmicsFusionFilteredSupplementary.csv doesnt exists")
            OmicsFusionFilteredSupplementary_df = pd.read_csv(filepath)
            return OmicsFusionFilteredSupplementary_df
        elif filename == "6_OmicsSomaticMutationsProfile":
            filepath = os.path.join(data_sub_path,"6_OmicsSomaticMutationsProfile.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 6_OmicsSomaticMutationsProfile.csv doesnt exists")
            OmicsSomaticMutationsProfile_df = pd.read_csv(filepath)
            return OmicsSomaticMutationsProfile_df
        elif filename == "7_cellosaurus":
            filepath = os.path.join(data_sub_path,"7_cellosaurus.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 7_cellosaurus.csv doesnt exists")
            cellosaurus_df = pd.read_csv(filepath)
            return cellosaurus_df
        elif filename == "8_DepMap_OmicsProfiles":
            filepath = os.path.join(data_sub_path,"8_DepMap_OmicsProfiles.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 8_DepMap_OmicsProfiles.csv doesnt exists")
            DepMap_OmicsProfiles_df = pd.read_csv(filepath)
            return DepMap_OmicsProfiles_df
        elif filename == "9_DepMap_sample_info":
            filepath = os.path.join(data_sub_path,"9_DepMap_sample_info.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 9_DepMap_sample_info.csv doesnt exists")
            DepMap_sample_info_df = pd.read_csv(filepath)
            return DepMap_sample_info_df
        elif filename == "10_GEOInfo":
            filepath = os.path.join(data_sub_path,"10_GEOInfo.txt")
            if not os.path.isfile(filepath):
                raise Exception("Data file 10_GEOInfo.txt doesnt exists")
            GEOInfo_df = pd.read_csv(filepath, sep='\t')
            return GEOInfo_df
        elif filename == "11_hpa_rna_celline_description":
            filepath = os.path.join(data_sub_path,"11_hpa_rna_celline_description.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 11_hpa_rna_celline_description.csv doesnt exists")
            hpa_rna_celline_description_df = pd.read_csv(filepath)
            return hpa_rna_celline_description_df
        elif filename == "12_CCLE_metabolomics_20190502":
            filepath = os.path.join(data_sub_path,"12_CCLE_metabolomics_20190502.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 12_CCLE_metabolomics_20190502.csv doesnt exists")
            CCLE_metabolomics_20190502_df = pd.read_csv(filepath)
            return CCLE_metabolomics_20190502_df
        elif filename == "13_CCLE_miRNA_20181103":
            filepath = os.path.join(data_sub_path,"13_CCLE_miRNA_20181103.gct")
            if not os.path.isfile(filepath):
                raise Exception("Data file 13_CCLE_miRNA_20181103.gct doesnt exists")
            CCLE_miRNA_20181103_df = pd.read_csv(filepath, sep='\t', skiprows=2)
            return CCLE_miRNA_20181103_df
        elif filename == "14_OmicsGlobalSignatures":
            filepath = os.path.join(data_sub_path,"14_OmicsGlobalSignatures.csv")
            if not os.path.isfile(filepath):
                raise Exception("Data file 14_OmicsGlobalSignatures.csv doesnt exists")
            OmicsGlobalSignatures_df = pd.read_csv(filepath)
            return OmicsGlobalSignatures_df
        elif filename == "mart_export":
            filepath = os.path.join(data_sub_path,"mart_export.txt")
            if not os.path.isfile(filepath):
                raise Exception("Data File mart_export.txt doesnt exists")
            mart_export_df = pd.read_csv(filepath, sep='\t')
            return mart_export_df
        else:
            raise Exception("Invalid Input given to the file name")
    except Exception as e:
        print(e)
        return None
    
def data_save(data_df, sudirectory, subfolder, filename):
    try:
        data_sub_path = os.path.join(base_path, "data", sudirectory, subfolder)
        if not os.path.isdir(data_sub_path):
            os.makedirs(data_sub_path, exist_ok=True)
        filepath = os.path.join(data_sub_path, filename)
        data_df.to_csv(filepath, index=False)
    except Exception as e:
        print(e)
        return None

def cleaned_data_reader(sudirectory, subfolder, filename):
    try:
        data_sub_path = os.path.join(base_path,"data", sudirectory, subfolder)
        if not os.path.isdir(data_sub_path):
            os.makedirs(data_sub_path, exist_ok=True)
        filepath = os.path.join(data_sub_path, filename)
        data_df = pd.read_csv(filepath)
        return data_df
    except Exception as e:
        print(e)
        return None
