import app.preprocessing as loader
import app.scoring as scoring
import app.mutation as mutate
import app.fusion as fuse
import app.modelloader as model
import numpy as np



def pipelinerun(targetgenelist, exclusiongenelist, diseasename, fusionfilter, mutationfilter):
    try:
        if targetgenelist is None:
            raise Exception("The Target Gene ID list is empty")
        exclusionchecker = False
        if exclusiongenelist is not None:
            exclusionchecker = True
        diseasechecker = False
        if diseasename is not None:
            diseasechecker = True

        target_esng = loader.genetoesng(targetgenelist)
        found_proteins, not_found_proteins = loader.esngtoprotein(target_esng)
        target_depmap_df = loader.depmapreader(target_esng)
        target_hpamap_df = loader.hparnareader(target_esng)
        target_harmonized_protein_df = loader.proteinreader(found_proteins)
        target_cosine_df = loader.similarityreader(target_esng)
        if not_found_proteins is not None:
            for prot in not_found_proteins:
                fallback_column_name = prot
                target_harmonized_protein_df[fallback_column_name] = np.nan
        target_df = target_depmap_df.merge(target_harmonized_protein_df, on='ModelID', how='left')
        target_df = target_df.merge(target_hpamap_df, on='ModelID', how='left')
        if exclusionchecker:
            exclusion_esng = loader.genetoesng(exclusiongenelist)
            exclusion_found_proteins, exclusion_not_found_proteins = loader.esngtoprotein(exclusion_esng)
            exclusion_depmap_df = loader.depmapreader(exclusion_esng)
            exclusion_hpamap_df = loader.hparnareader(exclusion_esng)
            exclusion_harmonized_protein_df = loader.proteinreader(exclusion_found_proteins)
            exlusion_cosine_df = loader.similarityreader(exclusion_esng)
            if exclusion_not_found_proteins is not None:
                for prot in exclusion_not_found_proteins:
                    fallback_column_name = prot
                    exclusion_harmonized_protein_df[fallback_column_name] = np.nan
            exclusion_df = exclusion_depmap_df.merge(exclusion_harmonized_protein_df, on='ModelID', how='left')
            exclusion_df = exclusion_df.merge(exclusion_hpamap_df, on='ModelID', how='left')
            if diseasechecker:
                requiredlistofdiseasemasterid = loader.diseasecheker(diseasename)
                exclusion_df = exclusion_df[exclusion_df['ModelID'].isin(requiredlistofdiseasemasterid)]
        
        if diseasechecker:
            requiredlistofdiseasemasterid = loader.diseasecheker(diseasename)
            target_df = target_df[target_df['ModelID'].isin(requiredlistofdiseasemasterid)]


        ## scoring
        target_evidence_df = scoring.calculateevidencescore(target_df.copy(), 'target')
        target_similarity_df = scoring.calculatesimilarityscore(target_cosine_df.copy(), 'target')
        if exclusionchecker:
            new_confidence_df = target_df.merge(exclusion_df, on='ModelID', how='left')
            net_confidence_df = scoring.calculateconfidencescore(new_confidence_df)
        else:
            new_confidence_df = target_df.copy()
            net_confidence_df = scoring.calculateconfidencescore(new_confidence_df)

        if exclusionchecker:
            exclusion_evidence_df = scoring.calculateevidencescore(exclusion_df.copy(), 'exclusion')
            exclusion_similarity_df = scoring.calculatesimilarityscore(exlusion_cosine_df.copy(), 'exclusion')

        if exclusionchecker:
            new_evidence_df = target_evidence_df.merge(exclusion_evidence_df, on='ModelID', how='left')
            net_evidence_df = scoring.calculatenetevidencescore(new_evidence_df.copy(), 'exclusion')
            new_similarity_df = target_similarity_df.merge(exclusion_similarity_df, on='ModelID', how='left')
            net_similarity_df = scoring.calculatenetsimilarityscore(new_similarity_df.copy(), 'exclusion')
        else:
            net_evidence_df = scoring.calculatenetevidencescore(target_evidence_df.copy(), 'target')
            net_similarity_df = scoring.calculatenetsimilarityscore(target_similarity_df.copy(), 'target')

        finalscoringdf = net_evidence_df.merge(net_confidence_df, on='ModelID', how='left')
        finalscoringdf = finalscoringdf.merge(net_similarity_df, on='ModelID', how='left')
        finalscoringdf = scoring.finalscoring(finalscoringdf.copy())
        if exclusionchecker:
            final_scored_df = target_df.merge(exclusion_df, on='ModelID', how='left')
            final_scored_df = final_scored_df.merge(finalscoringdf, on='ModelID', how='left')
        else:
            final_scored_df = target_df.copy()
            final_scored_df = final_scored_df.merge(finalscoringdf, on='ModelID', how='left')

        if exclusionchecker:
            geneslist = target_esng + exclusion_esng
            final_fusion_df = fuse.fusionflagger(final_scored_df, geneslist)
            if fusionfilter:
                final_fusion_df = fuse.remove_fusion_models(final_fusion_df)
        else:
            geneslist = target_esng
            final_fusion_df = fuse.fusionflagger(final_scored_df, geneslist)
            if fusionfilter:
                final_fusion_df = fuse.remove_fusion_models(final_fusion_df)
     
        if exclusionchecker:
            geneslist = target_esng + exclusion_esng
            final_mutated_df = mutate.mutationflager(final_fusion_df, geneslist)
            if mutationfilter:
                final_mutated_df = mutate.remove_mutation_models(final_mutated_df)
        else:
            geneslist = target_esng
            final_mutated_df = mutate.mutationflager(final_fusion_df, geneslist)
            if mutationfilter:
                final_mutated_df = mutate.remove_mutation_models(final_mutated_df)
        
        final_top10_df = final_mutated_df.sort_values('final_score', ascending=False)
        final_top10_df = final_top10_df.head(10)

        #get the reference tables

        if exclusionchecker:
            geneslist = target_esng + exclusion_esng
            if fusionfilter:
                fusion_reference_df = None
            else:
                fusion_reference_df = fuse.getthefusionreferencetable(final_top10_df, geneslist)
        else:
            geneslist = target_esng
            if fusionfilter:
                fusion_reference_df = None
            else:
                fusion_reference_df = fuse.getthefusionreferencetable(final_top10_df, geneslist)

        if exclusionchecker:
            geneslist = target_esng + exclusion_esng
            if mutationfilter:
                mutation_reference_df = None
            else:
                mutation_reference_df = mutate.getthemutationreferencetable(final_top10_df, geneslist)
        else:
            geneslist = target_esng
            if mutationfilter:
                mutation_reference_df = None
            else:
                mutation_reference_df = mutate.getthemutationreferencetable(final_top10_df, geneslist)

        final_top10_df, kmeansplotforselectedcell = model.kmeansonthedata(final_top10_df)
        alternative_df = model.knnonthedata(final_top10_df, final_mutated_df)

        final_top10_df = loader.datacombiner(final_top10_df)
        alternative_df = loader.alternativedatacombiner(alternative_df)
        return final_top10_df, alternative_df, fusion_reference_df, mutation_reference_df, kmeansplotforselectedcell, final_mutated_df
        
    except Exception as e:
        print(f"Error while running the pipeline: {e}")
        raise

