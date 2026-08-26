import ml_build.utils as ut
import pandas as pd
import numpy as np
import os
from ml_build.logger import get_logger

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
log = get_logger('Testing')

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def testingkmeansmodel():
    try:
        log.info("Starting KMeans Inference Validation Test")
        target_models = ['ACH-000005', 'ACH-0000011', 'ACH-000008', 'ACH-000015']
        log.info("Reading multi-omics latent factor matrix matrix")
        factor_df = ut.cleaned_data_reader('mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')
        if factor_df is None or factor_df.empty:
            raise Exception("Unable to fetch the factor matrix")
        testset = factor_df[factor_df["ModelID"].isin(target_models)]
        log.info(f"Target match tracking: Located {len(testset)} out of {len(target_models)} requested validation cell lines.")
        if testset.empty:
            raise ValueError(f"None of the target models {target_models} were found in the factor matrix")
        X = testset.drop(columns=["ModelID"]).values.astype("float64")
        log.info(f"KMeans inference matrix shape: {X.shape}")
        log.info("Loading serialized KMeans pipeline estimator instance")        
        model = ut.model_load('kmeans_model.joblib')
        if model is None:
            raise Exception("Model returned empty")
        log.info("Executing cluster coordinate predictions")
        labels = model.predict(X)
        cluster_df = pd.DataFrame({
            "ModelID": testset["ModelID"],
            "Cluster": labels
        })
        kmeans_metrics_path = os.path.join(base_path, 'artifacts', 'kmeanstestresults.txt')
        log.info(f"Writing KMeans cluster profiles out to: {kmeans_metrics_path}")
        os.makedirs(os.path.dirname(kmeans_metrics_path), exist_ok=True)
        with open(kmeans_metrics_path, "w") as f:
            f.write("KMeans test results\n")
            for i in range(len(cluster_df)):
                model_id = cluster_df.iloc[i]["ModelID"]
                cluster_num = cluster_df.iloc[i]["Cluster"]
                f.write(f"ModelID: {model_id} -> Cluster: {cluster_num}\n")
        log.info("KMeans inference validation completed successfully.")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the KMeans validation test block with error: {e}")
        raise


def testingknnmodel():
    try:
        log.info("Starting KNN Sister Neighborhood Selection Test")
        target_models = ['ACH-000005', 'ACH-0000011', 'ACH-000008', 'ACH-000015']
        log.info("Reading multi-omics latent factor matrix matrix")
        factor_df = ut.cleaned_data_reader('mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')
        if factor_df is None:
            raise Exception("Unable to fetch the factor matrix")
        testset = factor_df[factor_df["ModelID"].isin(target_models)]
        log.info(f"Target match tracking: Located {len(testset)} out of {len(target_models)} requested validation cell lines.")
        if testset.empty:
            raise ValueError(f"None of the target models {target_models} were found in the factor matrix.")
        X = testset.drop(columns=["ModelID"]).values.astype("float64")
        log.info(f"KNN neighbor query coordinates feature shape: {X.shape}")
        log.info("Loading serialized KNN neighbor graph mapping pipeline")
        model = ut.model_load('knn_model.joblib')
        if model is None:
            raise Exception("Model returned empty")
        log.info("Computing closest neighborhood indices and multi-dimensional distances")
        distances, indices = model.kneighbors(X)
        all_model_ids = factor_df["ModelID"].values
        target_ids_list = testset["ModelID"].tolist()
        knn_metrics_path = os.path.join(base_path, 'artifacts', 'knntestresults.txt')
        log.info(f"Writing alternative neighborhood assignments out to: {knn_metrics_path}")
        os.makedirs(os.path.dirname(knn_metrics_path), exist_ok=True)
        with open(knn_metrics_path, "w") as f:
            f.write("KNN Test Results (6 True Alternative Models)\n")
            f.write("--------------------------------------------------\n")
            for i, target_id in enumerate(target_ids_list):
                neighbor_rows = indices[i]
                all_7_models = all_model_ids[neighbor_rows]
                true_alternatives = all_7_models[1:]
                alt_models_str = ", ".join(true_alternatives)
                avg_alt_distance = np.mean(distances[i][1:])
                f.write(f"Target: {target_id} -> Avg Alt Distance: {avg_alt_distance:.4f} | Alternatives: [{alt_models_str}]\n")
        log.info("KNN neighborhood verification test complete.")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the KNN neighborhood test block with error: {e}")
        raise
