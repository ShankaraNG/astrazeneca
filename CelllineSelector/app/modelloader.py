import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import ml_build.utils as ut
import uuid
from app.logger import get_logger

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
log = get_logger('ModelLoader')
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kmeansonthedata(data_df):
    try:
        log.info(f"kmeansonthedata: {len(data_df)} selected cell line(s)")
        copied_input_df = data_df.copy()
        factor_df = ut.cleaned_data_reader('mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')
        if factor_df is None or factor_df.empty:
            raise ValueError("multiomincsfactor.csv is missing or empty.")
        chosen_ids = copied_input_df['ModelID'].tolist()
        kmeans = ut.model_load('kmeans_model.joblib')
        if kmeans is None:
            raise ValueError("Failed to load the KMeans model.")
        model_ids = factor_df["ModelID"].values
        X = factor_df.drop(columns=["ModelID"]).values.astype("float64")
        log.info(f"kmeansonthedata: factor matrix {X.shape}; predicting clusters")
        labels = kmeans.predict(X)
        log.info("kmeansonthedata: generating UMAP projection (full 1479-row manifold)")
        reducer = UMAP(n_neighbors=35, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X)
        viz_df = pd.DataFrame({
            'ModelID': model_ids,
            'UMAP_1': X_umap[:, 0],
            'UMAP_2': X_umap[:, 1],
            'Cluster': labels,
        })
        cluster_mapping = viz_df[['ModelID', 'Cluster']].rename(columns={'Cluster': 'biological_sub_group'})
        copied_input_df = copied_input_df.merge(cluster_mapping, on='ModelID', how='left')
        highlight_df = viz_df[viz_df['ModelID'].isin(chosen_ids)]
        log.info(f"kmeansonthedata: {len(highlight_df)} of {len(chosen_ids)} chosen id(s) located on the map")
        log.info("kmeansonthedata: releasing factor matrix and UMAP arrays from memory")
        del factor_df, X, X_umap, reducer, labels, model_ids, cluster_mapping
        gc.collect()
        plt.figure(figsize=(12, 9))
        scatter = plt.scatter(viz_df['UMAP_1'], viz_df['UMAP_2'], c=viz_df['Cluster'],
                              cmap='tab20', s=25, alpha=0.35, edgecolors='none')
        plt.colorbar(scatter, label="Cluster ID")
        if not highlight_df.empty:
            plt.scatter(highlight_df['UMAP_1'], highlight_df['UMAP_2'], color='red',
                        edgecolor='black', s=200, marker='s', linewidths=1.5, label='Highlighted Targets')
            for _, row in highlight_df.iterrows():
                plt.text(row['UMAP_1'] + 0.15, row['UMAP_2'] + 0.15, str(row['ModelID']),
                         fontsize=9, weight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))
        else:
            log.warning("kmeansonthedata: none of the chosen ids matched records in factor_df")
        plt.title(f"UMAP Target Selection Map (KMeans K={kmeans.n_clusters})", fontsize=14, pad=15)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        run_id = str(uuid.uuid4())
        kmeansplotforselectedcell = os.path.join(base_path, 'images', f'kmeansplotforselectedcell_{run_id}.png')
        os.makedirs(os.path.dirname(kmeansplotforselectedcell), exist_ok=True)
        plt.savefig(kmeansplotforselectedcell, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"kmeansonthedata: plot saved to {kmeansplotforselectedcell}")
        log.info("kmeansonthedata: releasing plotting frames from memory")
        del viz_df, highlight_df
        gc.collect()
        return copied_input_df, kmeansplotforselectedcell
    except Exception as e:
        log.error(f"Error in kmeansonthedata: {e}")
        raise


def knnonthedata(data_df, scored_data_df):
    try:
        log.info(f"knnonthedata: finding alternatives for {len(data_df)} selected cell line(s)")
        factor_df = ut.cleaned_data_reader('mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')
        if factor_df is None or factor_df.empty:
            raise ValueError("Factor data is missing or empty.")
        chosen_ids = data_df['ModelID'].tolist()
        knn = ut.model_load('knn_model.joblib')
        if knn is None:
            raise ValueError("Failed to load the KNN model.")
        all_model_ids = factor_df["ModelID"].values
        alternativemodel = []
        forthemodelid = []
        for target_id in chosen_ids:
            target_row = factor_df[factor_df["ModelID"] == target_id]
            if target_row.empty:
                log.warning(f"knnonthedata: {target_id} not in factor matrix; skipping")
                continue
            X_query = target_row.drop(columns=["ModelID"]).values.astype("float64")
            distances, indices = knn.kneighbors(X_query, n_neighbors=15)
            row_neighbor_indices = indices[0]
            tempalternative = []
            for neighbor_idx in row_neighbor_indices:
                neighbor_id = all_model_ids[neighbor_idx]
                if (neighbor_id != target_id and neighbor_id not in chosen_ids
                        and neighbor_id not in alternativemodel):
                    tempalternative.append(neighbor_id)
            if not tempalternative:
                alternativemodel.append(None)
                forthemodelid.append(target_id)
            else:
                selected_alts_df = scored_data_df[scored_data_df["ModelID"].isin(tempalternative)]
                if not selected_alts_df.empty:
                    best_alt_row = selected_alts_df.sort_values(by="final_score", ascending=False).iloc[0]
                    forthemodelid.append(target_id)
                    alternativemodel.append(best_alt_row['ModelID'])
                else:
                    alternativemodel.append(None)
                    forthemodelid.append(target_id)
        log.info("knnonthedata: releasing factor matrix from memory")
        del factor_df, all_model_ids
        gc.collect()
        result_df = pd.DataFrame({"ModelID": forthemodelid, "AlternativeModelID": alternativemodel})
        n_found = result_df['AlternativeModelID'].notna().sum()
        log.info(f"knnonthedata: {n_found} of {len(result_df)} target(s) received an alternative")
        return result_df
    except Exception as e:
        log.error(f"Error in knnonthedata: {e}")
        raise