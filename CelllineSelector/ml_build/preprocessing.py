from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from mofapy2.run.entry_point import entry_point
import ml_build.utils as ut
import numpy as np
from ml_build.logger import get_logger
import gc


log = get_logger('Preprocessing')

def preprocessingformodel(data_df):
    try:
        log.info("Starting standard scaling transformation (Z-score)")
        if data_df is None or (isinstance(data_df, pd.DataFrame) and data_df.empty):
            raise ValueError("Input dataset for standard scaling is null or empty.")
        scaler = StandardScaler()
        data_df = scaler.fit_transform(data_df)
        log.info("Standard scaling completed successfully.")
        return data_df
    except Exception as e:
        log.error(f"Error during standard scaling preprocessing: {e}")
        raise

def preprocessingforscoring(data_df):
    try:
        log.info("Starting MinMax scaling transformation (0-1 range)")
        if data_df is None or (isinstance(data_df, pd.DataFrame) and data_df.empty):
            raise ValueError("Input dataset for MinMax scaling is null or empty.")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_df)
        log.info("MinMax scaling completed successfully")
        return scaled_data
    except Exception as e:
        log.error(f"Error during MinMax scaling preprocessing: {e}")
        raise

##### Too Much memory consumption and failing

# def mofamatrixformodel(depmap_omicsexpressionallgenes_cleaned_df,harmonized_ms_celline_cleaned_df,CCLE_metabolomics_cleaned_20190502_df,OmicsGlobalSignatures_cleaned_df):
#     try:
#         log.info("Starting Multi-Omics MOFA Matrix Construction Step")
#         if depmap_omicsexpressionallgenes_cleaned_df is None or depmap_omicsexpressionallgenes_cleaned_df.empty:
#             raise ValueError("Required depmap_omicsexpressionallgenes_cleaned_df is missing or empty.")   
#         log.info("Sorting baseline expression records by ModelID numerical suffix")
#         depmap_omicsexpressionallgenes_cleaned_df = depmap_omicsexpressionallgenes_cleaned_df.sort_values(by="ModelID",key=lambda x: x.str.replace("ACH-", "", regex=False).astype(int)).reset_index(drop=True)
#         master_ids = depmap_omicsexpressionallgenes_cleaned_df["ModelID"]
#         log.info(f"Master index established with {len(master_ids)} unique ModelIDs")
#         log.info("Aligning multi-omic tracking views across master ModelID mapping space")
#         protein_df = (harmonized_ms_celline_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index())
#         metabolomics_df = (CCLE_metabolomics_cleaned_20190502_df.set_index("ModelID").reindex(master_ids).reset_index())
#         globalsig_df = (OmicsGlobalSignatures_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index())
#         log.info("Dropping non-feature metadata keys from omic dataframes")
#         globalsig_df = globalsig_df.drop(columns=['ModelConditionID', 'SequencingID', 'IsDefaultEntryForModel', 'IsDefaultEntryForMC'])
#         metabolomics_df = metabolomics_df.drop(columns=['CCLE_ID'])
#         log.info("Melting dataset arrays into long-format tracking series")
#         rna_long = depmap_omicsexpressionallgenes_cleaned_df.melt(
#             id_vars="ModelID",
#             var_name="feature",
#             value_name="value"
#         )

#         rna_long["view"] = "RNA"
#         rna_long["group"] = "CellLines"

#         protein_long = protein_df.melt(
#             id_vars="ModelID",
#             var_name="feature",
#             value_name="value"
#         )

#         protein_long["view"] = "Proteomics"
#         protein_long["group"] = "CellLines"

#         metabolomics_long = metabolomics_df.melt(
#             id_vars="ModelID",
#             var_name="feature",
#             value_name="value"
#         )

#         metabolomics_long["view"] = "Metabolomics"
#         metabolomics_long["group"] = "CellLines"

#         globalsig_long = globalsig_df.melt(
#             id_vars="ModelID",
#             var_name="feature",
#             value_name="value"
#         )

#         globalsig_long["view"] = "GlobalSignatures"
#         globalsig_long["group"] = "CellLines"
        
#         data_long = pd.concat(
#             [
#                 rna_long,
#                 protein_long,
#                 metabolomics_long,
#                 globalsig_long
#             ],
#             ignore_index=True
#         )

#         data_long = data_long.rename(
#             columns={
#                 "ModelID": "sample"
#             }
#         )
#         log.info("Concatenating tracking long arrays into single master entry point frame")
#         log.info("Initializing mofapy2 execution workspace configurations")
#         ent = entry_point()
#         ent.set_data_options(
#             scale_views=False,
#             scale_groups=False,
#             center_groups=True,
#             use_float32=True
#         )
#         ent.set_data_df(data_long)
#         ent.set_model_options(
#             factors=20,
#             spikeslab_factors=False,
#             spikeslab_weights=True,
#             ard_factors=True,
#             ard_weights=True
#         )

#         ent.set_train_options(
#             iter=1000,
#             convergence_mode="medium",
#             seed=42
#         )
#         log.info("Building MOFA model graph layers")
#         ent.build()
#         log.info("Running MOFA variational inference optimization loop")
#         ent.run()
#         log.info("Extracting latent space expectation values (Z node)")
#         factors = ent.model.nodes["Z"].getExpectation()
#         log.info(f"Extracted raw factors array structure shape: {factors.shape}")

#         if factors.shape[0] == len(master_ids):

#             factor_df = pd.DataFrame(
#                 factors,
#                 columns=[
#                     f"Factor{i+1}"
#                     for i in range(factors.shape[1])
#                 ]
#             )

#         else:
#             log.warning("Extracted factor matrix matrix dimensions are inverted. Applying transposition step.")
#             factor_df = pd.DataFrame(
#                 factors.T,
#                 columns=[
#                     f"Factor{i+1}"
#                     for i in range(factors.shape[0])
#                 ]
#             )
        
#         factor_df.insert(
#             0,
#             "ModelID",
#             master_ids.values
#         )
#         log.info(f"Final structured Multi-Omics factor matrix frame layout shape: {factor_df.shape}")
#         log.info("Saving multi-omics feature dataset artifact...")
#         result = ut.data_save(factor_df, 'mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')
#         if result is None or result != "successfull":
#             raise Exception("Failed to save multiomincsfactor.csv matrix file.")
#         log.info("Multi-omics factor extraction run completed successfully.")
#         return factor_df
#     except Exception as e:
#         log.error(f"The pipeline failed in the Multi-Omics MOFA execution run with error: {e}")
#         raise

# def mofamatrixforcosine(depmap_omicsexpressionallgenes_cleaned_df):
#     try:
#         log.info("Starting RNA View-Specific MOFA Feature Extraction")
#         if depmap_omicsexpressionallgenes_cleaned_df is None or depmap_omicsexpressionallgenes_cleaned_df.empty:
#             raise ValueError("The input depmap_omicsexpressionallgenes_cleaned_df matrix is null or empty.")
#         log.info("Sorting baseline expression records by ModelID numerical suffix")
#         depmap_omicsexpressionallgenes_cleaned_df = depmap_omicsexpressionallgenes_cleaned_df.sort_values(by="ModelID",key=lambda x: x.str.replace("ACH-", "", regex=False).astype(int)).reset_index(drop=True)
#         master_ids = depmap_omicsexpressionallgenes_cleaned_df["ModelID"]
#         depmap_omicsexpressionallgenes_cleaned_df = (
#             depmap_omicsexpressionallgenes_cleaned_df
#             .set_index("ModelID")
#             .reindex(master_ids)
#             .reset_index()
#         )
#         rna_view = depmap_omicsexpressionallgenes_cleaned_df.drop(columns=["ModelID"])
#         rna_view = rna_view.loc[:, ~rna_view.columns.duplicated()]
#         log.info(f"Deduplicated raw expression grid tracking dimensions: {rna_view.shape}")
#         log.info("Initializing mofapy2 single-view data matrix workspace")
#         ent_rna = entry_point()
#         ent_rna.set_data_matrix(
#             data=[
#                 [
#                     rna_view.values.copy()
#                 ]
#             ],
#             views_names=[
#                 "RNA"
#             ],
#             groups_names=[
#                 "AllSamples"
#             ]
#         )
#         ent_rna.set_data_options(
#             scale_views=False,
#             scale_groups=False,
#             center_groups=True,
#             use_float32=True
#         )
#         N_FACTORS = 20

#         ent_rna.set_model_options(
#             factors=N_FACTORS,
#             spikeslab_factors=False,
#             spikeslab_weights=True,
#             ard_factors=False,
#             ard_weights=True
#         )
#         ent_rna.set_train_options(
#             iter=1000,
#             convergence_mode="medium",
#             verbose=True,
#             seed=42
#         )
#         log.info("Building view-specific structural graph layers")
#         ent_rna.build()
#         log.info("Running single-view model optimization loop")
#         ent_rna.run()
#         log.info("Extracting view-specific feature loading weights (W node)")
#         w_rna = ent_rna.model.nodes["W"].getExpectation()[0]
#         log.info(f"Extracted weight array feature node matrix shape: {w_rna.shape}")
#         w_rna_df = pd.DataFrame(
#             w_rna,
#             index=rna_view.columns,
#             columns=[
#                 f"Factor{i+1}"
#                 for i in range(w_rna.shape[1])
#             ]
#         )
#         log.info(f"Final parsed RNA loading matrix data tracking shape: {w_rna_df.shape}")
#         log.info("Saving view-specific RNA factor artifact configuration matrix")
#         result = ut.data_save(w_rna_df, 'mofa_data', 'rnafactor', 'rnafactor.csv')
#         if result is None or result != "successfull":
#             raise Exception("Failed to save rnafactor.csv matrix file.")
#         log.info("Single-view factor model extraction run completed successfully.")
#         return w_rna_df
#     except Exception as e:
#         log.error(f"The pipeline failed in the RNA View MOFA execution run with error: {e}")
#         raise

# def mofamatrixformodel(depmap_omicsexpressionallgenes_cleaned_df, harmonized_ms_celline_cleaned_df, CCLE_metabolomics_cleaned_20190502_df, OmicsGlobalSignatures_cleaned_df):
#     try:
#         log.info("Starting Multi-Omics MOFA Matrix Construction Step")
#         if depmap_omicsexpressionallgenes_cleaned_df is None or depmap_omicsexpressionallgenes_cleaned_df.empty:
#             raise ValueError("Required depmap_omicsexpressionallgenes_cleaned_df is missing or empty.")
        
#         if "ProfileID" in depmap_omicsexpressionallgenes_cleaned_df.columns:
#             log.info("Dropping ProfileID metadata column from the modeling tracking matrix.")
#             depmap_omicsexpressionallgenes_cleaned_df = depmap_omicsexpressionallgenes_cleaned_df.drop(columns=["ProfileID"])        

#         log.info("Sorting baseline expression records by ModelID numerical suffix")
#         depmap_omicsexpressionallgenes_cleaned_df = depmap_omicsexpressionallgenes_cleaned_df.sort_values(
#             by="ModelID", key=lambda x: x.str.replace("ACH-", "", regex=False).astype(int)
#         ).reset_index(drop=True)
        
#         master_ids = depmap_omicsexpressionallgenes_cleaned_df["ModelID"]
#         log.info(f"Master index established with {len(master_ids)} unique ModelIDs")
        
#         log.info("Aligning multi-omic tracking views across master ModelID mapping space")
#         protein_df = (harmonized_ms_celline_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index())
#         metabolomics_df = (CCLE_metabolomics_cleaned_20190502_df.set_index("ModelID").reindex(master_ids).reset_index())
#         globalsig_df = (OmicsGlobalSignatures_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index())
        
#         log.info("Dropping non-feature metadata keys from omic dataframes")
#         globalsig_df = globalsig_df.drop(columns=['ModelConditionID', 'SequencingID', 'IsDefaultEntryForModel', 'IsDefaultEntryForMC'], errors='ignore')
#         metabolomics_df = metabolomics_df.drop(columns=['CCLE_ID'], errors='ignore')
        
#         #RAM optimization: Convert wide tables to float32 before melting to slash memory footprint by 50%
#         gene_features = [c for c in depmap_omicsexpressionallgenes_cleaned_df.columns if c != "ModelID"]
        
#         log.info("Melting dataset arrays into long-format tracking series")
#         rna_long = depmap_omicsexpressionallgenes_cleaned_df.melt(id_vars="ModelID", var_name="feature", value_name="value")
#         rna_long["view"], rna_long["group"] = "RNA", "CellLines"

#         protein_long = protein_df.melt(id_vars="ModelID", var_name="feature", value_name="value")
#         protein_long["view"], protein_long["group"] = "Proteomics", "CellLines"

#         metabolomics_long = metabolomics_df.melt(id_vars="ModelID", var_name="feature", value_name="value")
#         metabolomics_long["view"], metabolomics_long["group"] = "Metabolomics", "CellLines"

#         globalsig_long = globalsig_df.melt(id_vars="ModelID", var_name="feature", value_name="value")
#         globalsig_long["view"], globalsig_long["group"] = "GlobalSignatures", "CellLines"
        
#         data_long = pd.concat([rna_long, protein_long, metabolomics_long, globalsig_long], ignore_index=True)
#         data_long = data_long.rename(columns={"ModelID": "sample"})
        
#         # Free up the long pieces early
#         del rna_long, protein_long, metabolomics_long, globalsig_long
#         gc.collect()

#         log.info("Initializing mofapy2 execution workspace configurations")
#         ent = entry_point()
#         ent.set_data_options(scale_views=False, scale_groups=False, center_groups=True, use_float32=True)
#         ent.set_data_df(data_long)
#         ent.set_model_options(factors=20, spikeslab_factors=False, spikeslab_weights=True, ard_factors=True, ard_weights=True)
#         ent.set_train_options(iter=1000, convergence_mode="medium", seed=42)
        
#         log.info("Building MOFA model graph layers")
#         ent.build()
#         log.info("Running MOFA variational inference optimization loop")
#         ent.run()
        
#         # EXTRACT CELL LINE FACTORS (Z NODE)
#         log.info("Extracting latent space expectation values (Z node)")
#         factors = ent.model.nodes["Z"].getExpectation()
#         if factors.shape[0] != len(master_ids):
#             factors = factors.T
            
#         factor_df = pd.DataFrame(factors, columns=[f"Factor{i+1}" for i in range(factors.shape[1])])
#         factor_df.insert(0, "ModelID", master_ids.values)
#         ut.data_save(factor_df, 'mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')
        
#         # EXTRACT RNA FEATURE LOADING WEIGHTS (W NODE) DIRECTLY FROM THE CO-TRAINED MODEL
#         log.info("Extracting view-specific feature loading weights (W node) for RNA view")
#         # MOFA stores views indexed in order of training input or dictionary keys. Let's look up the index for 'RNA'
#         view_names = ent.model.view_names
#         rna_view_idx = view_names.index("RNA")
        
#         w_weights = ent.model.nodes["W"].getExpectation()[rna_view_idx]
        
#         w_rna_df = pd.DataFrame(w_weights, index=gene_features, columns=[f"Factor{i+1}" for i in range(w_weights.shape[1])])
#         log.info(f"Extracted RNA weight matrix layout shape: {w_rna_df.shape}")
#         ut.data_save(w_rna_df, 'mofa_data', 'rnafactor', 'rnafactor.csv')
        
#         log.info("Multi-omics factor and weight extraction run completed successfully.")
#         return factor_df, w_rna_df
#     except Exception as e:
#         log.error(f"The pipeline failed in the Multi-Omics MOFA execution run with error: {e}")
#         raise

def mofamatrixformodel(depmap_omicsexpressionallgenes_cleaned_df, harmonized_ms_celline_cleaned_df, CCLE_metabolomics_cleaned_20190502_df, OmicsGlobalSignatures_cleaned_df):
    try:
        log.info("Starting Multi-Omics MOFA Matrix Construction Step")
        if depmap_omicsexpressionallgenes_cleaned_df is None or depmap_omicsexpressionallgenes_cleaned_df.empty:
            raise ValueError("Required depmap_omicsexpressionallgenes_cleaned_df is missing or empty.")

        if "ProfileID" in depmap_omicsexpressionallgenes_cleaned_df.columns:
            log.info("Dropping ProfileID metadata column from the modeling tracking matrix.")
            depmap_omicsexpressionallgenes_cleaned_df = depmap_omicsexpressionallgenes_cleaned_df.drop(columns=["ProfileID"])

        log.info("Sorting baseline expression records by ModelID numerical suffix")
        depmap_omicsexpressionallgenes_cleaned_df = depmap_omicsexpressionallgenes_cleaned_df.sort_values(
            by="ModelID", key=lambda x: x.str.replace("ACH-", "", regex=False).astype(int)
        ).reset_index(drop=True)

        master_ids = depmap_omicsexpressionallgenes_cleaned_df["ModelID"]
        log.info(f"Master index established with {len(master_ids)} unique ModelIDs")

        log.info("Aligning multi-omic tracking views across master ModelID mapping space")
        protein_df = harmonized_ms_celline_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index()
        metabolomics_df = CCLE_metabolomics_cleaned_20190502_df.set_index("ModelID").reindex(master_ids).reset_index()
        globalsig_df = OmicsGlobalSignatures_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index()

        log.info("Dropping non-feature metadata keys from omic dataframes")
        globalsig_df = globalsig_df.drop(columns=['ModelConditionID', 'SequencingID', 'IsDefaultEntryForModel', 'IsDefaultEntryForMC'], errors='ignore')
        metabolomics_df = metabolomics_df.drop(columns=['CCLE_ID'], errors='ignore')

        # Build wide per-view feature blocks (drop ID col, dedup RNA columns)
        rna_view = depmap_omicsexpressionallgenes_cleaned_df.drop(columns=["ModelID"])
        rna_view = rna_view.loc[:, ~rna_view.columns.duplicated()]
        protein_view = protein_df.drop(columns=["ModelID"])
        metab_view = metabolomics_df.drop(columns=["ModelID"])
        globalsig_view = globalsig_df.drop(columns=["ModelID"])

        gene_features = list(rna_view.columns)  # aligns 1:1 with W rows below

        # float32 numpy matrices — MOFA reads these directly, no long-format melt
        rna_arr = rna_view.to_numpy(dtype=np.float32)
        protein_arr = protein_view.to_numpy(dtype=np.float32)
        metab_arr = metab_view.to_numpy(dtype=np.float32)
        globalsig_arr = globalsig_view.to_numpy(dtype=np.float32)

        log.info(f"View shapes -> RNA:{rna_arr.shape} Prot:{protein_arr.shape} Metab:{metab_arr.shape} GlobalSig:{globalsig_arr.shape}")

        views_names = ["RNA", "Proteomics", "Metabolomics", "GlobalSignatures"]

        log.info("Initializing mofapy2 execution workspace configurations")
        ent = entry_point()
        ent.set_data_matrix(
            data=[
                [rna_arr],
                [protein_arr],
                [metab_arr],
                [globalsig_arr],
            ],
            views_names=views_names,
            groups_names=["CellLines"],
            samples_names=[list(master_ids.values)],
            features_names=[
                gene_features,
                list(protein_view.columns),
                list(metab_view.columns),
                list(globalsig_view.columns),
            ],
        )
        ent.set_data_options(scale_views=False, scale_groups=False, center_groups=True, use_float32=True)
        ent.set_model_options(factors=20, spikeslab_factors=False, spikeslab_weights=True, ard_factors=True, ard_weights=True)
        ent.set_train_options(iter=1000, convergence_mode="medium", seed=42)

        # free the big arrays we no longer need before training
        del rna_arr, protein_arr, metab_arr, globalsig_arr
        del protein_df, metabolomics_df, globalsig_df, protein_view, metab_view, globalsig_view
        gc.collect()

        log.info("Building MOFA model graph layers")
        ent.build()
        log.info("Running MOFA variational inference optimization loop")
        ent.run()

        # CELL LINE FACTORS (Z)
        log.info("Extracting latent space expectation values (Z node)")
        factors = ent.model.nodes["Z"].getExpectation()
        if factors.shape[0] != len(master_ids):
            factors = factors.T
        factors = factors.astype("float64")
        factor_df = pd.DataFrame(factors, columns=[f"Factor{i+1}" for i in range(factors.shape[1])])
        factor_df.insert(0, "ModelID", master_ids.values)
        ut.data_save(factor_df, 'mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')

        # RNA FEATURE LOADINGS (W) from the co-trained model
        log.info("Extracting view-specific feature loading weights (W node) for RNA view")
        rna_view_idx = views_names.index("RNA")  # deterministic: we defined the order
        w_weights = ent.model.nodes["W"].getExpectation()[rna_view_idx]
        if w_weights.shape[0] != len(gene_features):
            raise ValueError(f"RNA weight rows ({w_weights.shape[0]}) != gene_features ({len(gene_features)}) - feature misalignment")
        w_weights = w_weights.astype("float64")
        w_rna_df = pd.DataFrame(w_weights, index=gene_features, columns=[f"Factor{i+1}" for i in range(w_weights.shape[1])])
        log.info(f"Extracted RNA weight matrix layout shape: {w_rna_df.shape}")
        ut.data_save(w_rna_df, 'mofa_data', 'rnafactor', 'rnafactor.csv')

        log.info("Multi-omics factor and weight extraction run completed successfully.")
        return factor_df, w_rna_df
    except Exception as e:
        log.error(f"The pipeline failed in the Multi-Omics MOFA execution run with error: {e}")
        raise