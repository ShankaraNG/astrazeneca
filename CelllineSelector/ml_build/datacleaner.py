import pandas as pd
import numpy as np
import re
import ml_build.utils as ut

def datacleanerfile(data_df, filenumber):
    try:
        if data_df is None:
            raise Exception("Empty dataset")
        filenumber = int(filenumber)
        if filenumber == 1:
            data_df = data_df[[ "Gene", "Gene name", "Cell line", "nTPM" ]]
            data_df["log_nTPM"] = np.log2( data_df["nTPM"] + 1 )
            hpa_pivot_df = data_df.pivot_table( index="Cell line", columns="Gene", values="log_nTPM" )
            hpa_pivot_df = hpa_pivot_df.reset_index()
            print(hpa_pivot_df.shape)
            print(hpa_pivot_df.head())
            DepMap_sample_info_df = ut.data_reader('nomenclature', '9_DepMap_sample_info')
            conversionofrnatable = DepMap_sample_info_df[['DepMap_ID', 'cell_line_name', 'stripped_cell_line_name']]
            newhparna_df = hpa_pivot_df.merge(conversionofrnatable, left_on='Cell line', right_on='cell_line_name', how='left')
            fallback_merge = hpa_pivot_df[newhparna_df['DepMap_ID'].isna()].merge(conversionofrnatable, left_on='Cell line', right_on='stripped_cell_line_name', how='left')
            newhparna_df.loc[newhparna_df['DepMap_ID'].isna(), 'DepMap_ID'] = fallback_merge['DepMap_ID'].values
            newhparna_df.drop(columns=['cell_line_name', 'stripped_cell_line_name'], errors='ignore', inplace=True)
            print("Remaining NaN DepMap_IDs:", newhparna_df['DepMap_ID'].isna().sum())
            print("NaN 'Cell line' entries:", newhparna_df['Cell line'].isna().sum())
            print("Duplicates present", newhparna_df['DepMap_ID'].duplicated().sum())
            print("Final Shape:", newhparna_df.shape)
            if 'DepMap_ID' in newhparna_df.columns:
                depmap_col = newhparna_df.pop('DepMap_ID')
                newhparna_df.insert(1, 'ModelID', depmap_col)
            ut.data_save(newhparna_df, 'cleaned_data', 'gene expression', '1_4_hpa_rna_celline.csv')
            return newhparna_df
        elif filenumber == 2:
            new_columns = []
            for col in data_df.columns:
                col = str(col).strip()
                match = re.search(r"ENSG\d+",col)
                if match:
                    new_columns.append(match.group())
                else:
                    new_columns.append(col)
            data_df.columns = new_columns
            data_df = data_df.rename(columns={'Unnamed: 0' : 'ProfileID'})
            print(data_df.shape)
            mart_export_df = ut.data_reader('mart_export', 'mart_export')
            proteincoding = mart_export_df[mart_export_df['Gene type'] == 'protein_coding']
            proteincodingvalid_ensg_ids = set(proteincoding["Gene stable ID"].astype(str).str.strip())
            def filtercolumns(x):
                keepcolumns = []
                keepcolumns.append(x.columns[0])
                for columns in x.columns[1:]:
                    if columns in proteincodingvalid_ensg_ids:
                        keepcolumns.append(columns)
                
                return keepcolumns

            columnstokeep = filtercolumns(data_df)
            data_df = data_df[columnstokeep]
            print(data_df.shape)
            DepMap_OmicsProfiles_df = ut.data_reader('nomenclature', '8_DepMap_OmicsProfiles')
            data_df = data_df.merge(DepMap_OmicsProfiles_df, on='ProfileID', how='left')
            priority_cols = ["ModelID", "ProfileID"]
            other_cols = [
                col
                for col in data_df.columns
                if col not in priority_cols
            ]
            data_df = data_df[priority_cols + other_cols]
            data_df = data_df.drop(columns=[ "ModelCondition", "Datatype", "WESKit"])
            data_df = data_df.drop_duplicates(subset=['ModelID'], keep='first')
            print(data_df.shape)
            ut.data_save(data_df, 'cleaned_data', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
            return data_df
        elif filenumber == 3:
            data_df.iloc[:, 1:] = np.log2(data_df.iloc[:, 1:] + 1 )
            print(data_df.shape)
            print(data_df.head(10))
            ut.data_save(data_df, 'cleaned_data', 'gene expression', '3_GEOexpression.csv')
            return data_df
        elif filenumber == 4:
            data_df = data_df.rename(columns={data_df.columns[0]: "ModelID"})
            new_columns = ["ModelID"]
            for col in data_df.columns[1:]:
                col = str(col).strip()
                protein_id = re.split(r"\s*\(", col)[0]
                new_columns.append(protein_id)
            data_df.columns = new_columns
            print(data_df.shape)
            print(data_df.head(5))
            ut.data_save(data_df, 'cleaned_data', 'gene expression', '4_harmonized_ms_cellline_cleaned.csv')
            return data_df
        elif filenumber == 5:
            data_df = data_df.drop(columns=["Unnamed: 0"])
            def split_gene_ens(value):
                value = str(value).strip()
                match = re.match(r"(.+?)\s*\((.*?)\)",value)
                if match:
                    gene_name = match.group(1).strip()
                    ens_raw = match.group(2).strip()
                    ens_match = re.search(r"(ENSG\d+)",ens_raw)
                    ensg_id = ens_match.group(1) if ens_match else None
                    return pd.Series([gene_name, ensg_id])
                return pd.Series([value, None])
            data_df[["gene1", "gene1_ENSG_ID"]] = data_df["gene1(ENS ID)"].apply(split_gene_ens)
            data_df[["gene2", "gene2_ENSG_ID"]] = data_df["gene2(ENS ID)"].apply(split_gene_ens)
            data_df = data_df.drop(columns=["gene1(ENS ID)","gene2(ENS ID)"])
            print(data_df.shape)
            ut.data_save(data_df, 'cleaned_data', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
            return data_df
        elif filenumber == 6:
            DepMap_OmicsProfiles_df = ut.data_reader('nomenclature', '8_DepMap_OmicsProfiles')
            data_df = data_df.merge(DepMap_OmicsProfiles_df, on='ProfileID', how='left')
            priority_cols = ["ProfileID", "ModelID", "EnsemblGeneID"]
            other_cols = [col for col in data_df.columns if col not in priority_cols]
            data_df = data_df[priority_cols + other_cols]
            data_df = data_df.drop(columns=["ModelCondition", "Datatype", "WESKit"])
            print("The duplicates present in the data with ModelID", data_df['ModelID'].duplicated().sum())
            print("The duplicates present in the data", data_df.duplicated().sum())
            print("The Null values present in the data with respect to modelid", data_df['ModelID'].isna().sum())
            print("The Null values present in the data with respect to profileid", data_df['ProfileID'].isna().sum())
            print(data_df.shape)
            ut.data_save(data_df, 'cleaned_data', 'gene properties', '6_OmicsSomaticMutationsProfile.csv')
            return data_df
        elif filenumber == 12:
            data_df = data_df.rename(columns={'DepMap_ID': 'ModelID'})
            ut.data_save(data_df, 'cleaned_data', 'non gene expression', '12_CCLE_metabolomics_20190502.csv')
            return data_df
        elif filenumber == 14:
            data_df = data_df.copy()
            data_df = data_df.drop(columns=['Unnamed: 0'])
            data_df = data_df[data_df['IsDefaultEntryForModel'] == 'Yes']
            print(data_df.shape)
            print(data_df.head(5))
            ut.data_save(data_df, 'cleaned_data', 'non gene expression', '14_OmicsGlobalSignatures.csv')
            return data_df
        else:
            raise Exception("unexpected file number")
    except Exception as e:
        print(e)
        raise Exception
