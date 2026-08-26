import pandas as pd
import numpy as np
import re
import gc
import ml_build.utils as ut
from ml_build.logger import get_logger

log = get_logger('Data Cleaner')


def datacleanerfile(data_df, filenumber):
    try:
        log.info(f"Starting Data Cleaning Process for File Number: {filenumber}")
        if data_df is None:
            raise Exception("Empty dataset")
        log.info(f"Initial raw dataset shape: {data_df.shape}")
        filenumber = int(filenumber)

        if filenumber == 1:
            data_df = data_df[["Gene", "Gene name", "Cell line", "nTPM"]]
            data_df["log_nTPM"] = np.log2(data_df["nTPM"] + 1).astype("float32")

            log.info("Reshaping data_df to map Cell lines against Genes")
            # Collapse any duplicate (Cell line, Gene) pairs first so unstack won't raise.
            # This reproduces pivot_table's default mean aggregation but as a fast groupby.
            dup_count = data_df.duplicated(subset=["Cell line", "Gene"]).sum()
            if dup_count > 0:
                log.info(f"Collapsing {dup_count} duplicate (Cell line, Gene) pairs via mean")
                data_df = data_df.groupby(["Cell line", "Gene"], as_index=False)["log_nTPM"].mean()

            hpa_pivot_df = (
                data_df.set_index(["Cell line", "Gene"])["log_nTPM"]
                .unstack("Gene")
                .reset_index()
            )
            hpa_pivot_df.columns.name = None
            log.info(f"Reshaped DataFrame shape: {hpa_pivot_df.shape}")

            # Free the multi-million-row long frame before the merges
            del data_df
            gc.collect()

            log.info("Reading nomenclature 9_DepMap_sample_info mapping data")
            DepMap_sample_info_df = ut.data_reader('nomenclature', '9_DepMap_sample_info')
            if DepMap_sample_info_df is None or DepMap_sample_info_df.empty:
                raise ValueError("DepMap_sample_info_df is null")
            conversionofrnatable = DepMap_sample_info_df[['DepMap_ID', 'RRID']]
            conversionofrnatable = conversionofrnatable.rename(columns={'RRID': 'Cellosaurus ID', 'DepMap_ID': 'ModelID'})
            hpa_rna_celline_description_df = ut.data_reader('nomenclature', '11_hpa_rna_celline_description')
            if hpa_rna_celline_description_df is None or hpa_rna_celline_description_df.empty:
                raise ValueError("hpa_rna_celline_description_df is null")
            hpa_rna_celline_conversion_df = hpa_rna_celline_description_df[['Cell line', 'Cellosaurus ID']]
            log.info("Executing primary merge via standard cell line names")
            newhparna_df = hpa_pivot_df.merge(hpa_rna_celline_conversion_df, on='Cell line', how='left')
            log.info("Executing Secondary merge to determine model id")
            newhparna_df = newhparna_df.merge(conversionofrnatable, on='Cellosaurus ID', how='left')
            log.info(f"Remaining NaN DepMap_IDs: {newhparna_df['ModelID'].isna().sum()}")
            log.info(f"NaN 'Cell line' entries: {newhparna_df['Cell line'].isna().sum()}")
            log.info(f"Duplicate values present in DepMap_ID: {newhparna_df['ModelID'].duplicated().sum()}")
            log.info(f"Final structured shape before saving: {newhparna_df.shape}")
            if 'Cell line' in newhparna_df.columns:
                depmap_col = newhparna_df.pop('Cell line')
                newhparna_df.insert(1, 'Cell line', depmap_col)
            if 'Cellosaurus ID' in newhparna_df.columns:
                depmap_col = newhparna_df.pop('Cellosaurus ID')
                newhparna_df.insert(1, 'Cellosaurus ID', depmap_col)
            if 'ModelID' in newhparna_df.columns:
                depmap_col = newhparna_df.pop('ModelID')
                newhparna_df.insert(1, 'ModelID', depmap_col)

            log.info("Saving processed HPA RNA data")
            result = ut.data_save(newhparna_df, 'cleaned_data', 'gene expression', '1_4_hpa_rna_celline.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 1_4_hpa_rna_celline.csv")
            log.info("File 1 successfully processed and verified")
            return newhparna_df
        elif filenumber == 2:
            log.info("Cleaning and parsing Ensembl IDs from headers")
            new_columns = []
            for col in data_df.columns:
                col = str(col).strip()
                match = re.search(r"ENSG\d+", col)
                if match:
                    new_columns.append(match.group())
                else:
                    new_columns.append(col)
            data_df.columns = new_columns
            data_df = data_df.rename(columns={'Unnamed: 0': 'ProfileID'})
            log.info(f"Ensembl parsed DataFrame shape: {data_df.shape}")
            mart_export_df = ut.data_reader('mart_export', 'mart_export')
            if mart_export_df is None or mart_export_df.empty:
                raise ValueError("mart_export_df is null")
            log.info("Filtering matrix to track protein coding genes using BioMart specs")
            proteincoding = mart_export_df[mart_export_df['Gene type'] == 'protein_coding']
            proteincodingvalid_ensg_ids = set(proteincoding["Gene stable ID"].astype(str).str.strip())
            def filtercolumns(x):
                keepcolumns = [x.columns[0]]
                for columns in x.columns[1:]:
                    if columns in proteincodingvalid_ensg_ids:
                        keepcolumns.append(columns)
                return keepcolumns
            columnstokeep = filtercolumns(data_df)
            data_df = data_df[columnstokeep]
            log.info(f"Protein-coding filtered shape: {data_df.shape}")

            log.info("Mapping profile records to unified ModelIDs")
            DepMap_OmicsProfiles_df = ut.data_reader('nomenclature', '8_DepMap_OmicsProfiles')
            if DepMap_OmicsProfiles_df is None or DepMap_OmicsProfiles_df.empty:
                raise ValueError("DepMap_OmicsProfiles_df is null")
            data_df = data_df.merge(DepMap_OmicsProfiles_df, on='ProfileID', how='left')
            priority_cols = ["ModelID", "ProfileID"]
            other_cols = [col for col in data_df.columns if col not in priority_cols]
            data_df = data_df[priority_cols + other_cols]
            data_df = data_df.drop(columns=["ModelCondition", "Datatype", "WESKit"])
            data_df = data_df.drop_duplicates(subset=['ModelID'], keep='first')
            log.info(f"Final unique ModelID table shape: {data_df.shape}")

            log.info("Saving processed DepMap expression data matrix")
            result = ut.data_save(data_df, 'cleaned_data', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv")
            log.info("File 2 successfully processed and verified")
            return data_df
        elif filenumber == 3:
            log.info("Applying element-wise log2 transformation matrix")
            data_df.iloc[:, 1:] = np.log2(data_df.iloc[:, 1:] + 1)
            log.info(f"Transformed matrix layout shape: {data_df.shape}")
            log.info("Saving processed GEO expression dataset")
            result = ut.data_save(data_df, 'cleaned_data', 'gene expression', '3_GEOexpression.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 3_GEOexpression.csv")
            log.info("File 3 successfully processed and verified.")
            return data_df
        elif filenumber == 4:
            data_df = data_df.rename(columns={data_df.columns[0]: "ModelID"})
            log.info("Stripping down complex multi-omic feature string suffixes")
            new_columns = ["ModelID"]
            for col in data_df.columns[1:]:
                col = str(col).strip()
                protein_id = re.split(r"\s*\(", col)[0]
                new_columns.append(protein_id)
            data_df.columns = new_columns
            log.info(f"Normalized column headers shape: {data_df.shape}")
            log.info("Saving processed Mass Spec data matrix")
            result = ut.data_save(data_df, 'cleaned_data', 'gene expression', '4_harmonized_ms_cellline_cleaned.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 4_harmonized_ms_cellline_cleaned.csv")
            log.info("File 4 successfully processed and verified.")
            return data_df
        elif filenumber == 5:
            data_df = data_df.drop(columns=["Unnamed: 0"])
            log.info("Parsing combined gene notation elements into component targets")
            # Vectorized parse: gene name = text before first '(' ; ENSG id = first ENSG\d+ inside.
            # Reproduces the old split_gene_ens row-wise logic without 184k Series allocations.
            for prefix in ["gene1", "gene2"]:
                src = data_df[f"{prefix}(ENS ID)"].astype(str)
                data_df[prefix] = src.str.split(r"\s*\(", n=1, regex=True).str[0].str.strip()
                data_df[f"{prefix}_ENSG_ID"] = src.str.extract(r"(ENSG\d+)", expand=False)
            data_df = data_df.drop(columns=["gene1(ENS ID)", "gene2(ENS ID)"])
            log.info(f"Parsed fusion records shape: {data_df.shape}")
            log.info("Saving filtered structural fusion matrix")
            result = ut.data_save(data_df, 'cleaned_data', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 5_OmicsFusionFilteredSupplementary.csv")
            log.info("File 5 successfully processed and verified.")
            return data_df
        elif filenumber == 6:
            log.info("Merging somatic mutational vectors to core profile indices")
            DepMap_OmicsProfiles_df = ut.data_reader('nomenclature', '8_DepMap_OmicsProfiles')
            data_df = data_df.merge(DepMap_OmicsProfiles_df, on='ProfileID', how='left')
            priority_cols = ["ProfileID", "ModelID", "EnsemblGeneID"]
            other_cols = [col for col in data_df.columns if col not in priority_cols]
            data_df = data_df[priority_cols + other_cols]
            data_df = data_df.drop(columns=["ModelCondition", "Datatype", "WESKit"])
            log.info(f"ModelID structural duplicates: {data_df['ModelID'].duplicated().sum()}")
            log.info(f"Total row duplicate signatures: {data_df.duplicated().sum()}")
            log.info(f"Null occurrences in ModelID tracking: {data_df['ModelID'].isna().sum()}")
            log.info(f"Null occurrences in ProfileID tracking: {data_df['ProfileID'].isna().sum()}")
            log.info(f"Final output shape: {data_df.shape}")
            log.info("Saving mutation profile configurations")
            result = ut.data_save(data_df, 'cleaned_data', 'gene properties', '6_OmicsSomaticMutationsProfile.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 6_OmicsSomaticMutationsProfile.csv")
            log.info("File 6 successfully processed and verified.")
            return data_df
        elif filenumber == 12:
            data_df = data_df.rename(columns={'DepMap_ID': 'ModelID'})
            log.info(f"Standardized metabolomics shape: {data_df.shape}")
            log.info("Saving baseline metabolomics array")
            result = ut.data_save(data_df, 'cleaned_data', 'non gene expression', '12_CCLE_metabolomics_20190502.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 12_CCLE_metabolomics_20190502.csv")
            log.info("File 12 successfully processed and verified.")
            return data_df

        elif filenumber == 14:
            data_df = data_df.copy()
            data_df = data_df.drop(columns=['Unnamed: 0'])
            log.info("Isolating records containing standard 'IsDefaultEntryForModel' flags")
            data_df = data_df[data_df['IsDefaultEntryForModel'] == 'Yes']
            log.info(f"Filtered default profile shape: {data_df.shape}")
            log.info("Saving standard multi-omics profile matrix")
            result = ut.data_save(data_df, 'cleaned_data', 'non gene expression', '14_OmicsGlobalSignatures.csv')
            if result is None or result != "successfull":
                raise Exception("Failed to save 14_OmicsGlobalSignatures.csv")
            log.info("File 14 successfully processed and verified.")
            return data_df
        else:
            raise Exception("unexpected file number")
    except Exception as e:
        log.error(f"The pipeline failed in the Data Cleaner execution for file {filenumber} with error: {e}")
        raise