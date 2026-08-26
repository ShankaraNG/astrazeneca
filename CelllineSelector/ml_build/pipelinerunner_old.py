import ml_build.utils as ut
import ml_build.datacleaner as datacleaner 
import ml_build.preprocessing as preprocessor
import ml_build.training as training
import ml_build.testing as testing
import ml_build.cosinematrix as cosine
import ml_build.utils as ut
from ml_build.logger import get_logger
import gc

log = get_logger('Pipelinerunner')

def modelpipeline(depmap_omicsexpressionallgenes_cleaned_df,harmonized_ms_celline_cleaned_df, CCLE_metabolomics_20190502_cleaned_df, OmicsGlobalSignatures_cleaned_df):
    try:
        # #data reading
        # depmap_omicsexpressionallgenes_df = ut.data_reader('gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile')
        # harmonized_ms_celline_df = ut.data_reader('gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted')
        # CCLE_metabolomics_20190502_df = ut.data_reader('non gene expression', '12_CCLE_metabolomics_20190502')
        # OmicsGlobalSignatures_df = ut.data_reader('non gene expression', '14_OmicsGlobalSignatures')
        # #cleaning
        # depmap_omicsexpressionallgenes_cleaned_df = datacleaner.datacleanerfile(depmap_omicsexpressionallgenes_df, 2)
        # harmonized_ms_celline_cleaned_df = datacleaner.datacleanerfile(harmonized_ms_celline_df, 4)
        # CCLE_metabolomics_20190502_cleaned_df = datacleaner.datacleanerfile(CCLE_metabolomics_20190502_df, 12)
        # OmicsGlobalSignatures_cleaned_df = datacleaner.datacleanerfile(OmicsGlobalSignatures_df, 14)
        #preprocessing
        log.info("Starting the model preprocessing for depmap_omicsexpressionallgenes")
        gene_cols = depmap_omicsexpressionallgenes_cleaned_df.columns[2:]
        depmap_omicsexpressionallgenes_cleaned_df[gene_cols] = preprocessor.preprocessingformodel(depmap_omicsexpressionallgenes_cleaned_df[gene_cols])
        log.info("Preprocessing for modelling has been completed for depmap_omicsexpressionallgenes")
        log.info("Starting the model preprocessing for CCLE_metabolomics_20190502")
        master_id_set = set(depmap_omicsexpressionallgenes_cleaned_df['ModelID'])
        CCLE_metabolomics_20190502_cleaned_df = CCLE_metabolomics_20190502_cleaned_df[CCLE_metabolomics_20190502_cleaned_df['ModelID'].isin(master_id_set)]
        metabolite_cols = CCLE_metabolomics_20190502_cleaned_df.columns[2:]
        CCLE_metabolomics_20190502_cleaned_df[metabolite_cols] = preprocessor.preprocessingformodel(CCLE_metabolomics_20190502_cleaned_df[metabolite_cols])
        log.info("Preprocessing for modelling has been completed for CCLE_metabolomics_20190502")
        log.info("Preprocessing for modelling has been completed for OmicsGlobalSignatures")
        OmicsGlobalSignatures_cleaned_df = OmicsGlobalSignatures_cleaned_df[OmicsGlobalSignatures_cleaned_df['ModelID'].isin(master_id_set)]
        signature_cols = ["MSIScore","LoHFraction","CIN","Ploidy","Aneuploidy"]
        OmicsGlobalSignatures_cleaned_df[signature_cols] = preprocessor.preprocessingformodel(OmicsGlobalSignatures_cleaned_df[signature_cols])
        log.info("Starting the model preprocessing for OmicsGlobalSignatures")
        #MOFA
        log.info("preparing the MOFA factor for the model")
        factor_df, w_rna_df = preprocessor.mofamatrixformodel(depmap_omicsexpressionallgenes_cleaned_df,harmonized_ms_celline_cleaned_df,CCLE_metabolomics_20190502_cleaned_df,OmicsGlobalSignatures_cleaned_df)
        if factor_df is None or factor_df.empty:
            raise ValueError("factor_df is null or empty and MOFA matrix generation failed.")
        if w_rna_df is None or w_rna_df.empty:
            raise ValueError("w_rna_df is null or empty and MOFA matrix generation failed.")
        log.info("MOFA Model preparation completed")
        #training
        log.info("Trainig the Kmeans model on MOFA factors")
        result = training.trainingkmeansmodel(factor_df)
        if result is None or result != "successfull":
            raise Exception("Failed during the Kmeans model training")
        log.info("Kmeans MOFA factor training completed")
        log.info("Training the KNN model on MOFA factors")
        result = training.trainingknnmodel(factor_df)
        if result is None or result != "successfull":
            raise Exception("Failed during the KNN model training")
        log.info("KNN MOFA factor training completed")
        #testing
        log.info("Testing the Kmeans model")
        result = testing.testingkmeansmodel()
        if result is None or result != "successfull":
            raise Exception("Failed during the Kmeans model testing")
        log.info("Testing completed for the KMeans model")
        log.info("Testing the KNN model")
        result = testing.testingknnmodel()
        if result is None or result != "successfull":
            raise Exception("Failed during the Kmeans model testing")
        log.info("Testing completed for the KNN model")
        #cosine
        log.info("Preparing the factor weight for Genes")
        # w_rna_df = preprocessor.mofamatrixforcosine(depmap_omicsexpressionallgenes_cleaned_df)
        # if w_rna_df is None or w_rna_df.empty:
        #     raise ValueError("w_rna_df is null or empty and MOFA matrix generation failed.")
        # log.info("Preparation of the factor weight completed")
        log.info("Running for the similarity matrix")
        similarity_df = cosine.cosinematrix(factor_df, w_rna_df)
        if similarity_df is None or similarity_df.empty:
            raise ValueError("similarity_df matrix is either empty or null")
        log.info("Similarity matrix run has been completed")
        log.info("Running the similarity metrix for the scoring")
        gene_cols = similarity_df.columns[1:]
        similarity_df[gene_cols] = preprocessor.preprocessingforscoring(similarity_df[gene_cols])
        log.info("Similarity matrix run for the scoring has been completed")
        log.info("Saving the Similarity matrix file")
        result = ut.data_save(similarity_df, 'scoring', 'cosine_matrix', 'cosinematrix.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("File has been saved successfully")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the model run with error: {e}")
        raise



def scoringpipeline(hpa_rna_cleaned_df,depmap_omicsexpressionallgenes_cleaned_df, geoexpression_cleaned_df, harmonized_ms_celline_cleaned_df, OmicsFusionFilteredSupplementary_cleaned_df, OmicsSomaticMutationsProfile_cleaned_df, CCLE_metabolomics_20190502_cleaned_df, OmicsGlobalSignatures_cleaned_df):
    try:
        # #data reading
        # hpa_rna_df = ut.data_reader('gene expression', '1_4_hpa_rna_celline')
        # depmap_omicsexpressionallgenes_df = ut.data_reader('gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile')
        # geoexpression_df = ut.data_reader('gene expression', '3_GEOexpression')
        # harmonized_ms_celline_df = ut.data_reader('gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted')
        # OmicsFusionFilteredSupplementary_df = ut.data_reader('gene properties', '5_OmicsFusionFilteredSupplementary')
        # OmicsSomaticMutationsProfile_df = ut.data_reader('gene properties', '6_OmicsSomaticMutationsProfile')
        # CCLE_metabolomics_20190502_df = ut.data_reader('non gene expression', '12_CCLE_metabolomics_20190502')
        # OmicsGlobalSignatures_df = ut.data_reader('non gene expression', '14_OmicsGlobalSignatures')
        # #cleaning
        # hpa_rna_cleaned_df = datacleaner.datacleanerfile(hpa_rna_df, 1)
        # depmap_omicsexpressionallgenes_cleaned_df = datacleaner.datacleanerfile(depmap_omicsexpressionallgenes_df, 2)
        # geoexpression_cleaned_df = datacleaner.datacleanerfile(geoexpression_df, 3)
        # harmonized_ms_celline_cleaned_df = datacleaner.datacleanerfile(harmonized_ms_celline_df, 4)
        # OmicsFusionFilteredSupplementary_cleaned_df = datacleaner.datacleanerfile(OmicsFusionFilteredSupplementary_df, 5)
        # OmicsSomaticMutationsProfile_cleaned_df = datacleaner.datacleanerfile(OmicsSomaticMutationsProfile_df, 6)
        # CCLE_metabolomics_20190502_cleaned_df = datacleaner.datacleanerfile(CCLE_metabolomics_20190502_df, 12)
        # OmicsGlobalSignatures_cleaned_df = datacleaner.datacleanerfile(OmicsGlobalSignatures_df, 14)
        #applyingminmax
        log.info("Preprocessing of the hpa_rna dataset for scoring started")
        master_id_set = set(depmap_omicsexpressionallgenes_cleaned_df['ModelID'])
        hpa_rna_cleaned_df = hpa_rna_cleaned_df[hpa_rna_cleaned_df['ModelID'].isin(master_id_set)]
        other_cols = [col for col in hpa_rna_cleaned_df.columns if col not in ['Cell line', 'ModelID']]
        hpa_rna_cleaned_df = hpa_rna_cleaned_df[['Cell line', 'ModelID'] + other_cols]
        numcolumns = hpa_rna_cleaned_df.columns[2:]
        hpa_rna_cleaned_df[numcolumns] = preprocessor.preprocessingforscoring(hpa_rna_cleaned_df[numcolumns])
        log.info("Preprocessing for the scoring of the hpa_rna dataset completed")
        log.info("Preprocessing of the depmap_omicsexpressionallgenes dataset for scoring started")
        gene_cols = depmap_omicsexpressionallgenes_cleaned_df.columns[2:]
        depmap_omicsexpressionallgenes_cleaned_df[gene_cols] = preprocessor.preprocessingforscoring(depmap_omicsexpressionallgenes_cleaned_df[gene_cols])
        log.info("Preprocessing for the scoring of the depmap_omicsexpressionallgenes dataset completed")
        log.info("Preprocessing of the geoexpression dataset for scoring started")
        numcolumns = geoexpression_cleaned_df.columns[1:]
        geoexpression_cleaned_df[numcolumns] = preprocessor.preprocessingforscoring(geoexpression_cleaned_df[numcolumns])
        log.info("Preprocessing for the scoring of the geoexpression dataset completed")
        log.info("Preprocessing of the harmonized_ms_celline dataset for scoring started")
        numcolumns = harmonized_ms_celline_cleaned_df.columns[1:]
        harmonized_ms_celline_cleaned_df[numcolumns] = preprocessor.preprocessingforscoring(harmonized_ms_celline_cleaned_df[numcolumns])
        log.info("Preprocessing for the scoring of the harmonized_ms_celline dataset completed")
        #save the datafile
        log.info("Saving the datafiles")
        log.info("Starting to save the processed 1_4_hpa_rna datafile for scoring")
        result = ut.data_save(hpa_rna_cleaned_df, 'scoring', 'gene expression', '1_4_hpa_rna_celline.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 1_4_hpa_rna datafile for scoring")
        log.info("Starting to save the processed 2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile datafile for scoring")
        result = ut.data_save(depmap_omicsexpressionallgenes_cleaned_df, 'scoring', 'gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile datafile for scoring")
        log.info("Starting to save the processed 3_GEOexpression datafile for scoring")        
        result = ut.data_save(geoexpression_cleaned_df, 'scoring', 'gene expression', '3_GEOexpression.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 3_GEOexpression datafile for scoring")
        log.info("Starting to save the processed 4_Harmonized_MS_CCLE_Gygi_subsetted datafile for scoring")         
        result = ut.data_save(harmonized_ms_celline_cleaned_df, 'scoring', 'gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 4_Harmonized_MS_CCLE_Gygi_subsetted datafile for scoring")
        log.info("Starting to save the processed 5_OmicsFusionFilteredSupplementary datafile for scoring") 
        result = ut.data_save(OmicsFusionFilteredSupplementary_cleaned_df, 'scoring', 'gene properties', '5_OmicsFusionFilteredSupplementary.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 5_OmicsFusionFilteredSupplementary datafile for scoring")
        log.info("Starting to save the processed 6_OmicsSomaticMutationsProfile datafile for scoring")
        result = ut.data_save(OmicsSomaticMutationsProfile_cleaned_df, 'scoring', 'gene properties', '6_OmicsSomaticMutationsProfile.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 6_OmicsSomaticMutationsProfile datafile for scoring")
        log.info("Starting to save the processed 12_CCLE_metabolomics_20190502 datafile for scoring")
        result = ut.data_save(CCLE_metabolomics_20190502_cleaned_df, 'scoring', 'non gene expression', '12_CCLE_metabolomics_20190502.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 12_CCLE_metabolomics_20190502 datafile for scoring")
        log.info("Starting to save the processed 14_OmicsGlobalSignatures datafile for scoring")
        result = ut.data_save(OmicsGlobalSignatures_cleaned_df, 'scoring', 'non gene expression', '14_OmicsGlobalSignatures.csv')
        if result is None or result != "successfull":
            raise Exception("Failed during the saving")
        log.info("Completed Saving the datafile 12_CCLE_metabolomics_20190502 datafile for scoring")
        log.info("Starting to save the processed 14_OmicsGlobalSignatures datafile for scoring")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the scoring run with error: {e}")
        raise

def pipelinerunner():
    try:
        #data reading
        log.info("Starting with the scoring pipeline")
        log.info("Reading the datafiles")
        log.info("Loading the datafile 1_4_hpa_rna_celline")
        hpa_rna_df = ut.data_reader('gene expression', '1_4_hpa_rna_celline')
        if hpa_rna_df is None or hpa_rna_df.empty:
            raise ValueError("hpa_rna_df is null")
        log.info("Loading of the datafile 1_4_hpa_rna_celline completed")
        log.info("Loading the datafile 2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile")
        depmap_omicsexpressionallgenes_df = ut.data_reader('gene expression', '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile')
        if depmap_omicsexpressionallgenes_df is None or depmap_omicsexpressionallgenes_df.empty:
            raise ValueError("2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile is null")
        log.info("Loading of the datafile 2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile completed")
        log.info("Loading the datafile 3_GEOexpression")
        geoexpression_df = ut.data_reader('gene expression', '3_GEOexpression')
        if geoexpression_df is None or geoexpression_df.empty:
            raise ValueError("3_GEOexpression is null")
        log.info("Loading of the datafile 3_GEOexpression completed")
        log.info("Loading the datafile 4_Harmonized_MS_CCLE_Gygi_subsetted")
        harmonized_ms_celline_df = ut.data_reader('gene expression', '4_Harmonized_MS_CCLE_Gygi_subsetted')
        if harmonized_ms_celline_df is None or harmonized_ms_celline_df.empty:
            raise ValueError("4_Harmonized_MS_CCLE_Gygi_subsetted is null")
        log.info("Loading of the datafile 4_Harmonized_MS_CCLE_Gygi_subsetted completed")
        log.info("Loading the datafile 5_OmicsFusionFilteredSupplementary")
        OmicsFusionFilteredSupplementary_df = ut.data_reader('gene properties', '5_OmicsFusionFilteredSupplementary')
        if OmicsFusionFilteredSupplementary_df is None or OmicsFusionFilteredSupplementary_df.empty:
            raise ValueError("5_OmicsFusionFilteredSupplementary is null")
        log.info("Loading of the datafile 5_OmicsFusionFilteredSupplementary completed")
        log.info("Loading the datafile 6_OmicsSomaticMutationsProfile")
        OmicsSomaticMutationsProfile_df = ut.data_reader('gene properties', '6_OmicsSomaticMutationsProfile')
        if OmicsSomaticMutationsProfile_df is None or OmicsSomaticMutationsProfile_df.empty:
            raise ValueError("6_OmicsSomaticMutationsProfile is null")
        log.info("Loading of the datafile 6_OmicsSomaticMutationsProfile completed")
        log.info("Loading the datafile 12_CCLE_metabolomics_20190502")
        CCLE_metabolomics_20190502_df = ut.data_reader('non gene expression', '12_CCLE_metabolomics_20190502')
        if CCLE_metabolomics_20190502_df is None or CCLE_metabolomics_20190502_df.empty:
            raise ValueError("12_CCLE_metabolomics_20190502 is null")
        log.info("Loading of the datafile 12_CCLE_metabolomics_20190502 completed")
        log.info("Loading the datafile 14_OmicsGlobalSignatures")
        OmicsGlobalSignatures_df = ut.data_reader('non gene expression', '14_OmicsGlobalSignatures')
        if OmicsGlobalSignatures_df is None or OmicsGlobalSignatures_df.empty:
            raise ValueError("14_OmicsGlobalSignatures is null")
        log.info("Loading of the datafile 14_OmicsGlobalSignatures completed")
        log.info("Loading of all the data file has been completed")
        #cleaning
        log.info("Starting the cleaning of the files")
        log.info("Starting the cleaning of hpa_rna files")
        hpa_rna_cleaned_df = datacleaner.datacleanerfile(hpa_rna_df, 1)
        log.info("Cleaning has been completed for hpa_rna")
        log.info("Starting the cleaning of depmap_omicsexpressionallgenes files")
        depmap_omicsexpressionallgenes_cleaned_df = datacleaner.datacleanerfile(depmap_omicsexpressionallgenes_df, 2)
        log.info("Cleaning has been completed for depmap_omicsexpressionallgenes")
        log.info("Starting the cleaning of geoexpression files")
        geoexpression_cleaned_df = datacleaner.datacleanerfile(geoexpression_df, 3)
        log.info("Cleaning has been completed for geoexpression")
        log.info("Starting the cleaning of harmonized_ms_celline files")
        harmonized_ms_celline_cleaned_df = datacleaner.datacleanerfile(harmonized_ms_celline_df, 4)
        log.info("Cleaning has been completed for harmonized_ms_celline")
        log.info("Starting the cleaning of OmicsFusionFilteredSupplementary files")
        OmicsFusionFilteredSupplementary_cleaned_df = datacleaner.datacleanerfile(OmicsFusionFilteredSupplementary_df, 5)
        log.info("Cleaning has been completed for OmicsFusionFilteredSupplementary")
        log.info("Starting the cleaning of OmicsSomaticMutationsProfile files")
        OmicsSomaticMutationsProfile_cleaned_df = datacleaner.datacleanerfile(OmicsSomaticMutationsProfile_df, 6)
        log.info("Cleaning has been completed for OmicsSomaticMutationsProfile")
        log.info("Starting the cleaning of CCLE_metabolomics_20190502 files")
        CCLE_metabolomics_20190502_cleaned_df = datacleaner.datacleanerfile(CCLE_metabolomics_20190502_df, 12)
        log.info("Cleaning has been completed for CCLE_metabolomics_20190502")
        log.info("Starting the cleaning of OmicsGlobalSignatures files")
        OmicsGlobalSignatures_cleaned_df = datacleaner.datacleanerfile(OmicsGlobalSignatures_df, 14)
        log.info("Cleaning has been completed for OmicsGlobalSignatures")
        log.info("Cleaning of all the files have been completed")
        log.info("Clearing uncleaned raw dataframes from memory cache")
        del hpa_rna_df, depmap_omicsexpressionallgenes_df, geoexpression_df, harmonized_ms_celline_df
        del OmicsFusionFilteredSupplementary_df, OmicsSomaticMutationsProfile_df, CCLE_metabolomics_20190502_df, OmicsGlobalSignatures_df
        gc.collect()
        depmap_for_model = depmap_omicsexpressionallgenes_cleaned_df.copy()
        metabolomics_for_model = CCLE_metabolomics_20190502_cleaned_df.copy()
        globalsig_for_model = OmicsGlobalSignatures_cleaned_df.copy()
        ms_celline_for_model = harmonized_ms_celline_cleaned_df.copy()
        log.info("Starting the Model Pipeline")
        result = modelpipeline(depmap_for_model,ms_celline_for_model, metabolomics_for_model, globalsig_for_model)
        if result is None or result != "successfull":
            raise Exception("Failed during the model pipeline run") 
        log.info("Model Pipeline completed successfully")
        log.info("Purging model pipeline datasets and collecting garbage")
        del depmap_for_model, metabolomics_for_model, globalsig_for_model, ms_celline_for_model
        gc.collect()  
        log.info("Starting the Scoring Pipeline")     
        result = scoringpipeline(hpa_rna_cleaned_df,depmap_omicsexpressionallgenes_cleaned_df, geoexpression_cleaned_df, harmonized_ms_celline_cleaned_df, OmicsFusionFilteredSupplementary_cleaned_df, OmicsSomaticMutationsProfile_cleaned_df, CCLE_metabolomics_20190502_cleaned_df, OmicsGlobalSignatures_cleaned_df)
        if result is None or result != "successfull":
            raise Exception("Failed during the scoring pipeline run")
        log.info("Scoring pipeline completed successfully")   
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the pipelinerunner run with error: {e}")
        raise