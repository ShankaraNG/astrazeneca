import ml_build.utils as ut
import ml_build.datacleaner as datacleaner
import ml_build.preprocessing as preprocessor
import ml_build.training as training
import ml_build.testing as testing
import ml_build.cosinematrix as cosine
import ml_build.lookuptable as lookuptable
from ml_build.logger import get_logger
import gc

log = get_logger('Pipelinerunner')

def clean_data_reader(subfolder, name, filenum):
    log.info(f"Loading raw datafile {name}")
    raw = ut.data_reader(subfolder, name)
    if raw is None or raw.empty:
        raise ValueError(f"{name} is null or empty")
    log.info(f"Cleaning datafile {name}")
    datacleaner.datacleanerfile(raw, filenum)
    del raw
    gc.collect()
    log.info(f"Cleaned and released {name} from memory")


def cleaningstage():
    try:
        log.info("Starting cleaning stage (one file at a time to bound memory)")
        clean_data_reader('gene expression', '1_4_hpa_rna_celline', 1)
        clean_data_reader('gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile', 2)
        clean_data_reader('gene expression', '3_GEOexpression', 3)
        clean_data_reader('gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted', 4)
        clean_data_reader('gene properties', '5_OmicsFusionFilteredSupplementary', 5)
        clean_data_reader('gene properties', '6_OmicsSomaticMutationsProfile', 6)
        clean_data_reader('non gene expression', '12_CCLE_metabolomics_20190502', 12)
        clean_data_reader('non gene expression', '14_OmicsGlobalSignatures', 14)
        log.info("Cleaning stage complete — all cleaned files persisted to disk")
        gc.collect()
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the cleaning stage with error: {e}")
        raise

def modelpipeline():
    try:
        log.info("Model: reading cleaned depmap expression from disk")
        depmap_df = ut.cleaned_data_reader('cleaned_data', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        if depmap_df is None or depmap_df.empty:
            raise ValueError("cleaned depmap expression is null or empty")
        log.info("Model: preprocessing depmap expression (z-score)")
        gene_cols = depmap_df.columns[2:]
        depmap_df[gene_cols] = preprocessor.preprocessingformodel(depmap_df[gene_cols])
        master_id_set = set(depmap_df['ModelID'])
        log.info("Model: reading cleaned harmonized MS (proteomics) from disk")
        ms_df = ut.cleaned_data_reader('cleaned_data', 'gene expression', '4_harmonized_ms_cellline_cleaned.csv')
        if ms_df is None or ms_df.empty:
            raise ValueError("cleaned harmonized MS is null or empty")
        log.info("Model: reading cleaned metabolomics from disk")
        metab_df = ut.cleaned_data_reader('cleaned_data', 'non gene expression', '12_CCLE_metabolomics_20190502.csv')
        if metab_df is None or metab_df.empty:
            raise ValueError("cleaned metabolomics is null or empty")
        metab_df = metab_df[metab_df['ModelID'].isin(master_id_set)]
        metabolite_cols = metab_df.columns[2:]
        metab_df[metabolite_cols] = preprocessor.preprocessingformodel(metab_df[metabolite_cols])
        log.info("Model: reading cleaned global signatures from disk")
        globalsig_df = ut.cleaned_data_reader('cleaned_data', 'non gene expression', '14_OmicsGlobalSignatures.csv')
        if globalsig_df is None or globalsig_df.empty:
            raise ValueError("cleaned global signatures is null or empty")
        globalsig_df = globalsig_df[globalsig_df['ModelID'].isin(master_id_set)]
        signature_cols = ["MSIScore", "LoHFraction", "CIN", "Ploidy", "Aneuploidy"]
        globalsig_df[signature_cols] = preprocessor.preprocessingformodel(globalsig_df[signature_cols])
        log.info("Model: preparing MOFA factors")
        factor_df, w_rna_df = preprocessor.mofamatrixformodel(depmap_df, ms_df, metab_df, globalsig_df)
        if factor_df is None or factor_df.empty:
            raise ValueError("factor_df is null or empty and MOFA matrix generation failed.")
        if w_rna_df is None or w_rna_df.empty:
            raise ValueError("w_rna_df is null or empty and MOFA matrix generation failed.")
        del depmap_df, ms_df, metab_df, globalsig_df
        gc.collect()
        log.info("Model: MOFA preparation completed; inputs released")
        log.info("Model: training KMeans on MOFA factors")
        result = training.trainingkmeansmodel(factor_df)
        if result is None or result != "successfull":
            raise Exception("Failed during the Kmeans model training")
        log.info("Model: training KNN on MOFA factors")
        result = training.trainingknnmodel(factor_df)
        if result is None or result != "successfull":
            raise Exception("Failed during the KNN model training")
        log.info("Model: testing KMeans")
        result = testing.testingkmeansmodel()
        if result is None or result != "successfull":
            raise Exception("Failed during the Kmeans model testing")
        log.info("Model: testing KNN")
        result = testing.testingknnmodel()
        if result is None or result != "successfull":
            raise Exception("Failed during the KNN model testing")
        log.info("Model: computing cosine similarity matrix")
        similarity_df = cosine.cosinematrix(factor_df, w_rna_df)
        if similarity_df is None or similarity_df.empty:
            raise ValueError("similarity_df matrix is either empty or null")
        del factor_df, w_rna_df
        gc.collect()
        log.info("Model: min-max scaling similarity matrix for scoring")
        gene_cols = similarity_df.columns[1:]
        similarity_df[gene_cols] = preprocessor.preprocessingforscoring(similarity_df[gene_cols])
        result = ut.data_save(similarity_df, 'scoring', 'cosine_matrix', 'cosinematrix.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving of the similarity matrix")
        del similarity_df
        gc.collect()
        log.info("Model pipeline completed successfully")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the model run with error: {e}")
        raise

def scoringpipeline():
    try:
        log.info("Scoring: reading depmap ModelIDs for master filter")
        depmap_ids_df = ut.cleaned_data_reader('cleaned_data', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        if depmap_ids_df is None or depmap_ids_df.empty:
            raise ValueError("cleaned depmap expression is null or empty")
        master_id_set = set(depmap_ids_df['ModelID'])
        del depmap_ids_df
        gc.collect()
        log.info("Scoring: processing hpa_rna")
        hpa_df = ut.cleaned_data_reader('cleaned_data', 'gene expression', '1_4_hpa_rna_celline.csv')
        if hpa_df is None or hpa_df.empty:
            raise ValueError("cleaned hpa_rna is null or empty")
        hpa_df = hpa_df[hpa_df['ModelID'].isin(master_id_set)]
        other_cols = [c for c in hpa_df.columns if c not in ['Cell line', 'ModelID', 'Cellosaurus ID']]
        hpa_df = hpa_df[['ModelID', 'Cell line', 'Cellosaurus ID'] + other_cols]
        numcolumns = hpa_df.columns[3:]
        hpa_df[numcolumns] = preprocessor.preprocessingforscoring(hpa_df[numcolumns])
        result = ut.data_save(hpa_df, 'scoring', 'gene expression', '1_4_hpa_rna_celline.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving of hpa_rna")
        del hpa_df
        gc.collect()
        log.info("Scoring: processing depmap expression")
        depmap_df = ut.cleaned_data_reader('cleaned_data', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        if depmap_df is None or depmap_df.empty:
            raise ValueError("cleaned depmap expression is null or empty")
        gene_cols = depmap_df.columns[2:]
        depmap_df[gene_cols] = preprocessor.preprocessingforscoring(depmap_df[gene_cols])
        result = ut.data_save(depmap_df, 'scoring', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving of depmap expression")
        del depmap_df
        gc.collect()
        log.info("Scoring: processing geoexpression")
        geo_df = ut.cleaned_data_reader('cleaned_data', 'gene expression', '3_GEOexpression.csv')
        if geo_df is None or geo_df.empty:
            raise ValueError("cleaned geoexpression is null or empty")
        numcolumns = geo_df.columns[1:]
        geo_df[numcolumns] = preprocessor.preprocessingforscoring(geo_df[numcolumns])
        result = ut.data_save(geo_df, 'scoring', 'gene expression', '3_GEOexpression.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving of geoexpression")
        del geo_df
        gc.collect()
        log.info("Scoring: processing harmonized MS")
        ms_df = ut.cleaned_data_reader('cleaned_data', 'gene expression', '4_harmonized_ms_cellline_cleaned.csv')
        if ms_df is None or ms_df.empty:
            raise ValueError("cleaned harmonized MS is null or empty")
        numcolumns = ms_df.columns[1:]
        ms_df[numcolumns] = preprocessor.preprocessingforscoring(ms_df[numcolumns])
        result = ut.data_save(ms_df, 'scoring', 'gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving of harmonized MS")
        del ms_df
        gc.collect()
        passthrough = [
            ('gene properties', '5_OmicsFusionFilteredSupplementary.csv', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv'),
            ('gene properties', '6_OmicsSomaticMutationsProfile.csv', 'gene properties', '6_OmicsSomaticMutationsProfile.csv'),
            ('non gene expression', '12_CCLE_metabolomics_20190502.csv', 'non gene expression', '12_CCLE_metabolomics_20190502.csv'),
            ('non gene expression', '14_OmicsGlobalSignatures.csv', 'non gene expression', '14_OmicsGlobalSignatures.csv'),
        ]
        for src_sub, src_name, dst_sub, dst_name in passthrough:
            log.info(f"Scoring: passing through {src_name}")
            df = ut.cleaned_data_reader('cleaned_data', src_sub, src_name)
            if df is None or df.empty:
                raise ValueError(f"cleaned {src_name} is null or empty")
            result = ut.data_save(df, 'scoring', dst_sub, dst_name)
            if result is None or result != "successfull":
                raise Exception(f"Failed during the saving of {dst_name}")
            del df
            gc.collect()
        log.info("Starting to build the look up table")
        master_lookup, gene_ensg_map, gene_protein_map = lookuptable.build_lookup_tables()
        if (master_lookup is None or master_lookup.empty or gene_ensg_map is None or gene_ensg_map.empty or gene_protein_map is None or gene_protein_map.empty):
            raise ValueError("Failed building lookup tables: One or more tables are empty or None")
        log.info("Look up table has been built successfully")
        log.info("Scoring pipeline completed successfully")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the scoring run with error: {e}")
        raise

def pipelinerunner():
    try:
        log.info("Starting the pipeline runner")
        result = cleaningstage()
        if result is None or result != "successfull":
            raise Exception("Failed during the cleaning stage")
        log.info("Starting the Model Pipeline")
        result = modelpipeline()
        if result is None or result != "successfull":
            raise Exception("Failed during the model pipeline run")
        log.info("Model Pipeline completed successfully")
        gc.collect()
        log.info("Starting the Scoring Pipeline")
        result = scoringpipeline()
        if result is None or result != "successfull":
            raise Exception("Failed during the scoring pipeline run")
        log.info("Scoring pipeline completed successfully")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the pipelinerunner run with error: {e}")
        raise