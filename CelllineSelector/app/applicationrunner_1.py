import gc
import numpy as np
import app.preprocessing as loader
import app.scoring as scoring
import app.mutation as mutate
import app.fusion as fuse
import app.modelloader as model
from app.logger import get_logger
import ml_build.utils as ut


log = get_logger('ApplicationRunner')


def pipelinerun(targetgenelist, exclusiongenelist, diseasename, fusionfilter, mutationfilter):
    try:
        log.info("Starting the pipeline run")
        if not targetgenelist:
            raise ValueError("The Target Gene list is empty")

        exclusionchecker = bool(exclusiongenelist)
        diseasechecker = bool(diseasename)
        log.info(f"exclusionchecker={exclusionchecker} and diseasechecker={diseasechecker}")
        requiredlistofdiseasemasterid = None
        if diseasechecker:
            requiredlistofdiseasemasterid = loader.diseasecheker(diseasename)
        log.info("Resolving target ids and reading target omic tables")
        target_esng = loader.genetoesng(targetgenelist)
        if not target_esng:
            raise ValueError("None of the target genes resolved to an ENSG id")
        found_proteins, not_found_proteins = loader.esngtoprotein(target_esng)
        target_depmap_df = loader.depmapreader(target_esng)
        target_hpamap_df = loader.hparnareader(target_esng)
        target_harmonized_protein_df = loader.proteinreader(found_proteins)
        target_cosine_df = loader.similarityreader(target_esng)
        for prot in (not_found_proteins or []):
            target_harmonized_protein_df[prot] = np.nan
        target_df = target_depmap_df.merge(target_harmonized_protein_df, on='ModelID', how='left')
        target_df = target_df.merge(target_hpamap_df, on='ModelID', how='left')
        log.info(f"target_df shape is {target_df.shape}")
        # Free the per-reader target frames now that their data lives in target_df
        log.info("Releasing target reader frames from memory")
        del target_depmap_df, target_hpamap_df, target_harmonized_protein_df
        gc.collect()
        exclusion_df = None
        exclusion_esng = []
        exlusion_cosine_df = None
        if exclusionchecker:
            log.info("Resolving exclusion ids and reading exclusion omic tables")
            exclusion_esng = loader.genetoesng(exclusiongenelist)
            exclusion_found_proteins, exclusion_not_found_proteins = loader.esngtoprotein(exclusion_esng)
            exclusion_depmap_df = loader.depmapreader(exclusion_esng)
            exclusion_hpamap_df = loader.hparnareader(exclusion_esng)
            exclusion_harmonized_protein_df = loader.proteinreader(exclusion_found_proteins)
            exlusion_cosine_df = loader.similarityreader(exclusion_esng)
            for prot in (exclusion_not_found_proteins or []):
                exclusion_harmonized_protein_df[prot] = np.nan
            exclusion_df = exclusion_depmap_df.merge(exclusion_harmonized_protein_df, on='ModelID', how='left')
            exclusion_df = exclusion_df.merge(exclusion_hpamap_df, on='ModelID', how='left')
            log.info(f"exclusion_df shape is {exclusion_df.shape}")
            # Free the per-reader exclusion frames now that their data lives in exclusion_df
            log.info("Releasing exclusion reader frames from memory")
            del exclusion_depmap_df, exclusion_hpamap_df, exclusion_harmonized_protein_df
            gc.collect()
        if diseasechecker:
            log.info(f"Applying disease filter of ({len(requiredlistofdiseasemasterid)} on ModelIDs)")
            target_df = target_df[target_df['ModelID'].isin(requiredlistofdiseasemasterid)]
            if exclusionchecker:
                exclusion_df = exclusion_df[exclusion_df['ModelID'].isin(requiredlistofdiseasemasterid)]
            if target_df.empty:
                raise ValueError(f"Disease filter '{diseasename}' left no target cell lines")
        log.info("Scoring: evidence + similarity (target)")
        target_evidence_df = scoring.calculateevidencescore(target_df.copy(), 'target')
        target_similarity_df = scoring.calculatesimilarityscore(target_cosine_df.copy(), 'target')
        # target_cosine_df is consumed; release it
        log.info("Releasing target cosine frame from memory")
        del target_cosine_df
        gc.collect()
        log.info("Calculating the confidence score")
        if exclusionchecker:
            overlap = (set(target_df.columns) & set(exclusion_df.columns)) - {'ModelID'}
            if overlap:
                log.warning(f"Confidence merge: {len(overlap)} overlapping column name(s) "
                            f"between target and exclusion; suffixing to avoid collision: {sorted(overlap)}")
                new_confidence_df = target_df.merge(exclusion_df, on='ModelID', how='left', suffixes=('_tgt', '_excl'))
            else:
                new_confidence_df = target_df.merge(exclusion_df, on='ModelID', how='left')
        else:
            new_confidence_df = target_df.copy()
        net_confidence_df = scoring.calculateconfidencescore(new_confidence_df)
        # new_confidence_df is consumed by the confidence score; release it
        log.info("Releasing confidence input frame from memory")
        del new_confidence_df
        gc.collect()
        if exclusionchecker:
            log.info("Scoring: evidence and similarity (exclusion)")
            exclusion_evidence_df = scoring.calculateevidencescore(exclusion_df.copy(), 'exclusion')
            exclusion_similarity_df = scoring.calculatesimilarityscore(exlusion_cosine_df.copy(), 'exclusion')
            # exlusion_cosine_df is consumed; release it
            log.info("Releasing exclusion cosine frame from memory")
            del exlusion_cosine_df
            gc.collect()
            new_evidence_df = target_evidence_df.merge(exclusion_evidence_df, on='ModelID', how='left')
            net_evidence_df = scoring.calculatenetevidencescore(new_evidence_df.copy(), 'exclusion')
            new_similarity_df = target_similarity_df.merge(exclusion_similarity_df, on='ModelID', how='left')
            net_similarity_df = scoring.calculatenetsimilarityscore(new_similarity_df.copy(), 'exclusion')
            log.info("Releasing intermediate evidence/similarity frames from memory")
            del exclusion_evidence_df, exclusion_similarity_df, new_evidence_df, new_similarity_df
            gc.collect()
        else:
            net_evidence_df = scoring.calculatenetevidencescore(target_evidence_df.copy(), 'target')
            net_similarity_df = scoring.calculatenetsimilarityscore(target_similarity_df.copy(), 'target')
        # target evidence/similarity frames are folded into the net frames; release
        log.info("Releasing target evidence/similarity frames from memory")
        del target_evidence_df, target_similarity_df
        gc.collect()
        log.info("Assembling final score (net_similarity x net_evidence x confidence)")
        finalscoringdf = net_evidence_df.merge(net_confidence_df, on='ModelID', how='left')
        finalscoringdf = finalscoringdf.merge(net_similarity_df, on='ModelID', how='left')
        finalscoringdf = scoring.finalscoring(finalscoringdf.copy())
        log.info("Releasing net score component frames from memory")
        del net_evidence_df, net_confidence_df, net_similarity_df
        gc.collect()
        # attach the score columns back onto the target (+exclusion) feature frame
        if exclusionchecker:
            final_scored_df = target_df.merge(exclusion_df, on='ModelID', how='left',suffixes=('_tgt', '_excl'))
        else:
            final_scored_df = target_df.copy()
        final_scored_df = final_scored_df.merge(finalscoringdf, on='ModelID', how='left')
        geneslist = target_esng + exclusion_esng if exclusionchecker else target_esng
        # target_df / exclusion_df / finalscoringdf are folded into final_scored_df; release
        log.info("Releasing feature and final-score frames from memory")
        del finalscoringdf, target_df
        if exclusionchecker:
            del exclusion_df
        gc.collect()
        log.info(f"Fusion flagging over {len(geneslist)} gene(s); filter={fusionfilter}")
        final_fusion_df = fuse.fusionflagger(final_scored_df, geneslist)
        # final_scored_df is superseded by final_fusion_df; release
        del final_scored_df
        gc.collect()
        if fusionfilter:
            final_fusion_df = fuse.remove_fusion_models(final_fusion_df)
        log.info(f"Mutation flagging over {len(geneslist)} gene(s); filter={mutationfilter}")
        final_mutated_df = mutate.mutationflager(final_fusion_df, geneslist)
        # final_fusion_df is superseded by final_mutated_df; release
        del final_fusion_df
        gc.collect()
        if mutationfilter:
            final_mutated_df = mutate.remove_mutation_models(final_mutated_df)
        if 'final_score' not in final_mutated_df.columns:
            raise KeyError("final_score column missing before ranking; scoring assembly failed")
        final_top10_df = final_mutated_df.sort_values('final_score', ascending=False).head(10)
        log.info(f"Selected top {len(final_top10_df)} cell line(s)")
        if fusionfilter:
            fusion_reference_df = None
        else:
            fusion_reference_df = fuse.getthefusionreferencetable(final_top10_df, geneslist)
        if mutationfilter:
            mutation_reference_df = None
        else:
            mutation_reference_df = mutate.getthemutationreferencetable(final_top10_df, geneslist)

        log.info("Running KMeans plot")
        final_top10_df, kmeansplotforselectedcell = model.kmeansonthedata(final_top10_df)
        log.info("Kmeans plot completed")
        log.info("Running KNN")
        alternative_df = model.knnonthedata(final_top10_df, final_mutated_df)
        log.info("KNN Completed")
        log.info("Running Data combiner")
        final_top10_df = loader.datacombiner(final_top10_df)
        alternative_df = loader.alternativedatacombiner(alternative_df)
        log.info("Data combiner completed")
        log.info("pipelinerun completed successfully")
        return final_top10_df, alternative_df, fusion_reference_df, mutation_reference_df, kmeansplotforselectedcell, final_mutated_df
    
    except Exception as e:
        log.error(f"Error while running the application pipeline: {e}")
        raise



#### Test run


targetgenelist = ['MTMR7', 'FKBP4', 'GGCT']
exclusiongenelist = ['UPP2']
diseasename = None
fusionfilter = False
mutationfilter = False
final_top10_df, alternative_df, fusion_reference_df, mutation_reference_df, kmeansplotforselectedcell, final_mutated_df = pipelinerun(targetgenelist, exclusiongenelist, diseasename, fusionfilter, mutationfilter)
ut.data_save(final_top10_df, 'test', 'top10', 'final_top10_df.csv')
ut.data_save(alternative_df, 'test', 'top10', 'alternative_top_10.csv')
ut.data_save(fusion_reference_df, 'test', 'top10', 'fusion_reference_df.csv')
ut.data_save(mutation_reference_df, 'test', 'top10', 'mutation_reference_df.csv')
ut.data_save(final_mutated_df, 'test', 'top10', 'scored_df.csv')