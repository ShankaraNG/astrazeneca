from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ml_build.utils as ut
from ml_build.logger import get_logger

log = get_logger('Cosine Matrix')

def cosinematrix(factor_df, w_rna_df):
    try:
        log.info("Starting the cosine matrix formation to find similarity")
        log.info(str(factor_df.shape))
        log.info(str(w_rna_df.shape))
        if factor_df is None or factor_df.empty:
            raise ValueError("factor_df is empty")
        if w_rna_df is None or w_rna_df.empty:
            raise ValueError("w_rna_df is empty")
        newfactor_df = factor_df.set_index('ModelID')
        newfactor_df.index.name = None
        Z = newfactor_df.values
        W = w_rna_df.values
        log.info("Starting to get the cosine values")
        similarity_matrix = cosine_similarity(Z, W)
        log.info("Starting to get the cosine values have been determined")
        log.info("Creating a dataframe")
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=newfactor_df.index,
            columns=w_rna_df.index
        )
        log.info("Dataframe has been created")
        log.info(str(similarity_df.shape))
        similarity_df = similarity_df.sort_index(axis=1)
        similarity_df = similarity_df.reset_index().rename(columns={'index': 'ModelID'})
        log.info("Saving the raw cosine dataframe")
        result = ut.data_save(similarity_df, 'cosine_matrix', 'cosine', 'cosinematrix.csv')
        if result is None or result != "successfull":
            raise Exception("Failed to save the similarity dataset")
        log.info("Similarity dataset has been saved successfully returning the file")
        return similarity_df
    except Exception as e:
        log.error(f"The pipeline failed in the Cosine Matrix run with error: {e}")
        raise