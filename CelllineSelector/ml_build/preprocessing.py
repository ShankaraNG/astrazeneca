from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from mofapy2.run.entry_point import entry_point
import ml_build.utils as ut

def preprocessingformodel(data_df):
    try:
        scaler = StandardScaler()
        data_df = scaler.fit_transform(data_df)
        return data_df
    except Exception as e:
        print(e)
        raise

def preprocessingforscoring(data_df):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_df)
        return scaled_data
    except Exception as e:
        print(e)
        raise

def mofamatrixformodel(depmap_omicsexpressionallgenes_cleaned_df,harmonized_ms_celline_cleaned_df,CCLE_metabolomics_cleaned_20190502_df,OmicsGlobalSignatures_cleaned_df):
    try:
        depmap_omicsexpressionallgenes_cleaned_df = depmap_omicsexpressionallgenes_cleaned_df.sort_values(by="ModelID",key=lambda x: x.str.replace("ACH-", "", regex=False).astype(int)).reset_index(drop=True)
        master_ids = depmap_omicsexpressionallgenes_cleaned_df["ModelID"]
        protein_df = (harmonized_ms_celline_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index())
        metabolomics_df = (CCLE_metabolomics_cleaned_20190502_df.set_index("ModelID").reindex(master_ids).reset_index())
        globalsig_df = (OmicsGlobalSignatures_cleaned_df.set_index("ModelID").reindex(master_ids).reset_index())
        globalsig_df = globalsig_df.drop(columns=['ModelConditionID', 'SequencingID', 'IsDefaultEntryForModel', 'IsDefaultEntryForMC'])
        metabolomics_df = metabolomics_df.drop(columns=['CCLE_ID'])
        rna_long = depmap_omicsexpressionallgenes_cleaned_df.melt(
            id_vars="ModelID",
            var_name="feature",
            value_name="value"
        )

        rna_long["view"] = "RNA"
        rna_long["group"] = "CellLines"

        protein_long = protein_df.melt(
            id_vars="ModelID",
            var_name="feature",
            value_name="value"
        )

        protein_long["view"] = "Proteomics"
        protein_long["group"] = "CellLines"

        metabolomics_long = metabolomics_df.melt(
            id_vars="ModelID",
            var_name="feature",
            value_name="value"
        )

        metabolomics_long["view"] = "Metabolomics"
        metabolomics_long["group"] = "CellLines"

        globalsig_long = globalsig_df.melt(
            id_vars="ModelID",
            var_name="feature",
            value_name="value"
        )

        globalsig_long["view"] = "GlobalSignatures"
        globalsig_long["group"] = "CellLines"

        data_long = pd.concat(
            [
                rna_long,
                protein_long,
                metabolomics_long,
                globalsig_long
            ],
            ignore_index=True
        )

        data_long = data_long.rename(
            columns={
                "ModelID": "sample"
            }
        )
        ent = entry_point()
        ent.set_data_options(
            scale_views=False,
            scale_groups=False,
            center_groups=True,
            use_float32=True
        )
        ent.set_data_df(data_long)
        ent.set_model_options(
            factors=20,
            spikeslab_factors=False,
            spikeslab_weights=True,
            ard_factors=True,
            ard_weights=True
        )

        ent.set_train_options(
            iter=1000,
            convergence_mode="medium",
            seed=42
        )
        ent.build()
        ent.run()
        factors = ent.model.nodes["Z"].getExpectation()

        print("Factors shape:", factors.shape)

        if factors.shape[0] == len(master_ids):

            factor_df = pd.DataFrame(
                factors,
                columns=[
                    f"Factor{i+1}"
                    for i in range(factors.shape[1])
                ]
            )

        else:

            factor_df = pd.DataFrame(
                factors.T,
                columns=[
                    f"Factor{i+1}"
                    for i in range(factors.shape[0])
                ]
            )

        factor_df.insert(
            0,
            "ModelID",
            master_ids.values
        )
        ut.data_save(factor_df, 'mofa_data', 'multiomincsfactor', 'multiomincsfactor.csv')
        return factor_df
    except Exception as e:
        print(e)
        raise