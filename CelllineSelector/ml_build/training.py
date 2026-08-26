import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import ml_build.utils as ut
from umap import UMAP
import os
from ml_build.logger import get_logger

log = get_logger('Training')

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def trainingkmeansmodel(factor_df):
    try:
        log.info("Starting KMeans Unsupervised Clustering Routine")
        artifacts_dir = os.path.join(base_path, 'artifacts')
        if not os.path.exists(artifacts_dir):
            log.info(f"Target directory {artifacts_dir} not found. Constructing fresh folder path.")
            os.makedirs(artifacts_dir, exist_ok=True)
        else:
            log.info(f"Target verification complete: Directory {artifacts_dir} already exists.")
        X = factor_df.drop(columns=["ModelID"]).values
        log.info(f"Extracted feature grid array shape for KMeans training: {X.shape}")
        def kmeansbestvalue(X):
            kmeans_results = []
            method = "MOFA"
            for k in range(2, 50):
                kmeans = KMeans(n_clusters=k,random_state=42)
                labels = kmeans.fit_predict(X)
                kmeans_results.append({
                    "Method": method,
                    "K": k,
                    "CalinskiHarabasz": calinski_harabasz_score(X, labels),
                    "DaviesBouldin": davies_bouldin_score(X, labels),
                    "Inertia": kmeans.inertia_,
                    "Silhouette": silhouette_score(X, labels),
                })
            kmeans_results_df = pd.DataFrame(kmeans_results)
            plt.figure(figsize=(8,5))
            plt.plot(kmeans_results_df["K"], kmeans_results_df["Inertia"],marker="o")
            plt.title(f"MOFA Inertia")
            plt.xlabel("K")
            plt.ylabel("Inertia")
            plt.grid(True)
            kmeansintertia = os.path.join(base_path, 'artifacts', 'kmeansintertia.png')
            plt.savefig(kmeansintertia, dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(8,5))
            plt.plot(kmeans_results_df["K"], kmeans_results_df["Silhouette"],marker="o")
            plt.title(f"MOFA Silhouette")
            plt.xlabel("K")
            plt.ylabel("Silhouette")
            plt.grid(True)
            kmeansSilhouette = os.path.join(base_path, 'artifacts', 'kmeansSilhouette.png')
            plt.savefig(kmeansSilhouette, dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(8,5))
            plt.plot(kmeans_results_df["K"], kmeans_results_df["CalinskiHarabasz"],marker="o")
            plt.title(f"MOFA CalinskiHarabasz")
            plt.xlabel("K")
            plt.ylabel("CalinskiHarabasz")
            plt.grid(True)
            kmeansCalinskiHarabasz = os.path.join(base_path, 'artifacts', 'kmeansCalinskiHarabasz.png')
            plt.savefig(kmeansCalinskiHarabasz, dpi=300, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(8,5))
            plt.plot(kmeans_results_df["K"], kmeans_results_df["DaviesBouldin"],marker="o")
            plt.title(f"MOFA DaviesBouldin")
            plt.xlabel("K")
            plt.ylabel("DaviesBouldin")
            plt.grid(True)
            kmeansDaviesBouldin = os.path.join(base_path, 'artifacts', 'kmeansDaviesBouldin.png')
            plt.savefig(kmeansDaviesBouldin, dpi=300, bbox_inches='tight')
            plt.close()
            return kmeans_results_df
        log.info("Running the Kmeans to select the best value of the K")
        log.info("Running the Kmeans from 2 to 50")
        kmeansresults = kmeansbestvalue(X)
        best_row = kmeansresults.sort_values(by="Silhouette", ascending=False).iloc[0]
        bestkvalue = int(best_row["K"])
        log.info(f"Best value of k is {bestkvalue}")
        log.info(f"Best silhouette value of K is {best_row['Silhouette']}")
        log.info(f"Best Inertia value of K is {best_row['Inertia']}")
        log.info(f"Initializing KMeans execution block with K={bestkvalue}")
        kmeans = KMeans(n_clusters=bestkvalue,random_state=42)
        labels = kmeans.fit_predict(X)
        log.info("Saving serialized KMeans pipeline estimator binary artifact")
        result = ut.model_save(kmeans, 'kmeans_model.joblib')
        if result is None or result != "successfull":
            raise Exception("Failed to Save the model")
        cluster_df = pd.DataFrame({
            "ModelID": factor_df["ModelID"],
            "Cluster": labels
        })
        log.info("Saving classified cluster label allocations dataframe...")
        result_save = ut.data_save(cluster_df, 'kmeans_data', 'kmeans', 'Kmeansclusters.csv')
        if result_save is None or result_save != "successfull":
            raise Exception("Failed to save Kmeansclusters.csv mapping matrix.")
        log.info("Calculating evaluation clustering performance metrics")
        Inertia = kmeans.inertia_
        Silhouette = silhouette_score(X, labels)
        kmeans_metrics_path = os.path.join(base_path, 'artifacts', 'kmeans_metrics.txt')
        log.info(f"Writing statistical cluster validation scores to: {kmeans_metrics_path}")
        with open(kmeans_metrics_path, "w") as f:
            f.write(f"KMeans Clustering Metrics (K={bestkvalue})\n")
            f.write("----------------------------------------\n")
            f.write(f"Inertia: {Inertia:.4f}\n")
            f.write(f"Silhouette Score: {Silhouette:.4f}\n")
        
        log.info("Generating raw latent feature distribution scatter plots")
        plt.figure(figsize=(10,8))
        plt.scatter(
            X[:,0],
            X[:,1],
            c=labels
        )
        plt.title(
            f"KMeans Best Clusters (K={bestkvalue})"
        )
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")
        kmeans_raw_factors_path = os.path.join(base_path, 'artifacts', 'kmeans_raw_factors.png')
        plt.savefig(kmeans_raw_factors_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info("Initializing UMAP manifold dimension reduction to visualize cluster boundaries")
        reducer = UMAP(n_neighbors=35, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_umap[:, 0], 
            X_umap[:, 1], 
            c=labels, 
            cmap='tab20',
            s=25,
            alpha=0.8
        )
        plt.title(f"UMAP Projection of KMeans Clusters (K={bestkvalue})")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.colorbar(scatter, label="Cluster ID")
        kmean_umap_factors_path = os.path.join(base_path, 'artifacts', 'kmeans_umap_projection.png')
        log.info(f"Saving UMAP 2D projection scatter plots to: {kmean_umap_factors_path}")
        plt.savefig(kmean_umap_factors_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info("KMeans model training and diagnostic artifact pipeline run complete.")
        return "successfull"
    except Exception as e:
        log.error(f"The pipeline failed in the KMeans training execution run with error: {e}")
        raise


def trainingknnmodel(factor_df):
    try:
        log.info("Starting KNN Neighborhood Graph Mapping Routine")
        artifacts_dir = os.path.join(base_path, 'artifacts')
        if not os.path.exists(artifacts_dir):
            log.info(f"Target directory {artifacts_dir} not found. Constructing fresh folder path.")
            os.makedirs(artifacts_dir, exist_ok=True)
        X = factor_df.drop(columns=["ModelID"]).values
        log.info(f"Extracted feature grid array shape for KNN fitting: {X.shape}")
        log.info("Running Kmeans to find the best value of K")
        def knnbestk(X):
            knn_results = []
            method = "MOFA"
            for k in range(2, 50):
                knn = NearestNeighbors(n_neighbors=k)
                knn.fit(X)
                distances, _ = knn.kneighbors(X)
                knn_results.append({
                    "Method": method,
                    "K": k,
                    "AverageDistance": np.mean(distances)})
                
            knn_results_df = pd.DataFrame(knn_results)
            plt.figure(figsize=(8,5))
            plt.plot(knn_results_df["K"],knn_results_df["AverageDistance"],marker="o")
            plt.title(f"{method} KNN Average Distance")
            plt.xlabel("K")
            plt.ylabel("Distance")
            plt.grid(True)
            KNN_Average_Distance = os.path.join(base_path, 'artifacts', 'KNN_Average_Distance.png')
            plt.savefig(KNN_Average_Distance, dpi=300, bbox_inches='tight')
            plt.close()
        knnbestk(X)
        log.info("Fitting mathematical unsupervised graph structure with K=6")
        knn = NearestNeighbors(n_neighbors=6)
        knn.fit(X)
        distances, _ = knn.kneighbors(X)
        avg_distances = np.mean(distances, axis=1)
        log.info("Saving serialized KNN graph matrix state block")
        result = ut.model_save(knn, 'knn_model.joblib')
        if result is None or result != "successfull":
            raise Exception("Failed to Save the model")
        knn_df = pd.DataFrame({
            "ModelID": factor_df["ModelID"],
            "AverageDistance": avg_distances
        })
        log.info("Saving average neighbor neighborhood distance vector records")
        result_save = ut.data_save(knn_df, 'knn_data', 'knn', 'KnnDistances.csv')
        if result_save is None or result_save != "successfull":
            raise Exception("Failed to save KnnDistances.csv metric file.")
        overall_mean_distance = np.mean(distances)
        knn_metrics_path = os.path.join(base_path, 'artifacts', 'knn_metrics.txt')
        log.info(f"Writing global neighborhood closeness baseline scores to: {knn_metrics_path}")
        with open(knn_metrics_path, "w") as f:
            f.write(f"KNN Model Metrics (K=6)\n")
            f.write("----------------------------------------\n")
            f.write(f"Overall Average Distance: {overall_mean_distance:.4f}\n")
        log.info("Plotting neighborhood local density scatter spaces")
        plt.figure(figsize=(10, 8))
        scatter1 = plt.scatter(
            X[:, 0],
            X[:, 1],
            c=avg_distances,
            cmap='viridis',
            alpha=0.8
        )
        plt.title("KNN Raw Factors (K=6)")
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")
        plt.colorbar(scatter1, label="Average Distance to Neighbors")
        knn_raw_factors_path = os.path.join(base_path, 'artifacts', 'knn_raw_factors.png')
        plt.savefig(knn_raw_factors_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info("Constructing decision boundary mesh space for 2D geographic region mapping")
        reducer = UMAP(n_neighbors=35, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        scatter2 = plt.scatter(
            X_umap[:, 0], 
            X_umap[:, 1], 
            c=avg_distances, 
            cmap='viridis',
            s=25,
            alpha=0.8
        )
        plt.title("UMAP Projection of KNN Distances (K=6)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.colorbar(scatter2, label="Average Distance to Neighbors")
        knn_umap_factors_path = os.path.join(base_path, 'artifacts', 'knn_umap_projection.png')
        plt.savefig(knn_umap_factors_path, dpi=300, bbox_inches='tight')
        plt.close()
        reducer = UMAP(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        
        knn = NearestNeighbors(n_neighbors=6)
        knn.fit(X_2d)
        
        point_ids = np.arange(len(X_2d))
        h = 0.1
        x_min, x_max = X_2d[:,0].min()-1, X_2d[:,0].max()+1
        y_min, y_max = X_2d[:,1].min()-1, X_2d[:,1].max()+1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        distances, indices = knn.kneighbors(grid_points, n_neighbors=1)
        Z = point_ids[indices.flatten()]
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(12,8))
        plt.contourf(xx, yy, Z, alpha=0.35, cmap='viridis') # 'tab20' handles discrete IDs well
        plt.scatter(
            X_2d[:,0],
            X_2d[:,1],
            s=25,
            edgecolor="black",
            c='#2077b4'
        )
        plt.title(f"KNN Neighborhood Regions for K=6")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.grid(True)
        knn_umap_area_map = os.path.join(base_path, 'artifacts', 'knn_umap_area_map.png')
        plt.savefig(knn_umap_area_map, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"Saving 2D spatial region area contour projection map to: {knn_umap_area_map}")
        log.info("KNN neighbor model training, mapping, and metric logs complete.")
        return "successfull"     
    except Exception as e:
        log.error(f"The pipeline failed in the KNN neighborhood training block with error: {e}")
        raise



