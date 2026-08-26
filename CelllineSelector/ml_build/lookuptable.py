import os
import re
import numpy as np
import pandas as pd
from ml_build.logger import get_logger
import ml_build.utils as ut

log = get_logger('LookupTable')

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def clean_name(name):
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())

def save_csv(df, subfolder, filename):
    try:
        log.info(f"Saving artifact '{filename}' (shape={df.shape}) to subfolder '{subfolder}'")
        out_dir = os.path.join(base_path, 'data', 'lookup', subfolder)
        if not os.path.isdir(out_dir):
            log.info(f"Output directory does not exist, creating: {out_dir}")
            os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, filename + '.csv')

        df_csv = df.copy()
        for col in df_csv.columns:
            if df_csv[col].apply(lambda v: isinstance(v, list)).any():
                log.info(f"Column '{col}' holds list values; stringifying for CSV export")
                df_csv[col] = df_csv[col].astype(str)
        df_csv.to_csv(csv_path, index=False)
        log.info(f"Successfully saved '{filename}' to: {csv_path}")
        return "successfull"
    except Exception as e:
        log.error(f"Failed to save artifact '{filename}' to disk: {e}")
        raise

def build_master_lookup():
    try:
        log.info("Building master_lookup")

        log.info("Reading nomenclature source files for master lookup")
        depmap_profiles = ut.data_reader('nomenclature', '8_DepMap_OmicsProfiles')
        sample_info = ut.data_reader('nomenclature', '9_DepMap_sample_info')
        geo_info = ut.data_reader('nomenclature', '10_GEOInfo')
        cellosaurus = ut.data_reader('nomenclature', '7_cellosaurus')
        for name, df in [('8_DepMap_OmicsProfiles', depmap_profiles),
                         ('9_DepMap_sample_info', sample_info),
                         ('10_GEOInfo', geo_info),
                         ('7_cellosaurus', cellosaurus)]:
            if df is None or df.empty:
                raise ValueError(f"{name} is null or empty")
            log.info(f"Loaded {name}: shape={df.shape}")
        log.info("Establishing anchor set from RNA-profiled ACH- codes")
        rna_profiles = (depmap_profiles[depmap_profiles['Datatype'] == 'rna'] 
                        [['ProfileID', 'ModelID']].copy())
        rna_profiles.columns = ['PR_ID', 'ACH_ID']
        before_dedup = len(rna_profiles)
        rna_profiles = rna_profiles.drop_duplicates(subset='ACH_ID', keep='first')
        log.info(f"Anchor ACH- codes: {len(rna_profiles)} unique "
                 f"(dropped {before_dedup - len(rna_profiles)} duplicate PR- per ACH-)")

        log.info("Attaching sample_info identity columns (names, CVCL, disease, lineage)")
        sample_cols = sample_info[['DepMap_ID', 'cell_line_name', 'stripped_cell_line_name',
                                   'CCLE_Name', 'RRID', 'primary_disease', 'lineage']].copy()
        sample_cols.columns = ['ACH_ID', 'cell_line_name', 'stripped_cell_line_name',
                               'CCLE_Name', 'CVCL_ID', 'primary_disease', 'lineage']
        master = rna_profiles.merge(sample_cols, on='ACH_ID', how='left')
        log.info(f"Master frame after identity merge: shape={master.shape}")
        log.info(f"Missing cell_line_name: {master['cell_line_name'].isna().sum()} "
                 f"| Missing CVCL_ID: {master['CVCL_ID'].isna().sum()}")
        log.info("Building ACH- -> name map from Cellosaurus cross-references")
        cvcl_name_map = {}
        for _, row in cellosaurus.iterrows():
            cross_ref = str(row.get('Cross-references', ''))
            if 'ACH-' in cross_ref:
                for ach in re.findall(r'ACH-\d+', cross_ref):
                    cvcl_name_map[ach] = row['Identifier (cell line name)']
        log.info(f"ACH- codes found in Cellosaurus cross-references: {len(cvcl_name_map)}")
        log.info("Resolving display names via priority chain "
                 "(sample_info -> CCLE_Name -> Cellosaurus -> stripped -> ACH_ID)")

        def resolve_name(row):
            if pd.notna(row['cell_line_name']) and str(row['cell_line_name']).strip() != '':
                return row['cell_line_name']
            if pd.notna(row['CCLE_Name']):
                return str(row['CCLE_Name']).split('_')[0]
            cvcl = cvcl_name_map.get(row['ACH_ID'])
            if cvcl:
                return cvcl
            if pd.notna(row['stripped_cell_line_name']):
                return row['stripped_cell_line_name']
            return row['ACH_ID']

        master['cell_line_name_resolved'] = master.apply(resolve_name, axis=1)
        anonymous_ach = master.loc[
            master['cell_line_name_resolved'] == master['ACH_ID'], 'ACH_ID'
        ].tolist()
        master['is_anonymous'] = master['ACH_ID'].isin(anonymous_ach)
        log.info(f"Name resolution complete: resolved={(~master['is_anonymous']).sum()} "
                 f"| anonymous (name == ACH_ID)={master['is_anonymous'].sum()}")

        log.info("Selecting and renaming final master columns")
        master_clean = master[['ACH_ID', 'PR_ID', 'CVCL_ID', 'CCLE_Name',
                               'cell_line_name_resolved', 'stripped_cell_line_name',
                               'primary_disease', 'lineage', 'is_anonymous']].copy()
        master_clean = master_clean.rename(columns={'cell_line_name_resolved': 'cell_line_name'})
        log.info("Back-filling missing CVCL IDs from Cellosaurus name maps")
        missing_before = master_clean['CVCL_ID'].isna().sum()
        cvcl_by_name = (cellosaurus.dropna(subset=['Identifier (cell line name)'])
                        .set_index('Identifier (cell line name)')['Accession (CVCL_xxxx)']
                        .to_dict())
        cvcl_by_clean = {clean_name(k): v for k, v in cvcl_by_name.items()}

        def resolve_cvcl(row):
            if pd.notna(row['CVCL_ID']):
                return row['CVCL_ID']
            cvcl = cvcl_by_name.get(row['cell_line_name'])
            if cvcl:
                return cvcl
            cvcl = cvcl_by_clean.get(clean_name(str(row['cell_line_name'])))
            if cvcl:
                return cvcl
            cvcl = cvcl_by_clean.get(clean_name(str(row['stripped_cell_line_name'])))
            if cvcl:
                return cvcl
            return np.nan

        master_clean['CVCL_ID'] = master_clean.apply(resolve_cvcl, axis=1)
        missing_after = master_clean['CVCL_ID'].isna().sum()
        log.info(f"CVCL back-fill: missing before={missing_before} -> after={missing_after} "
                 f"(recovered {missing_before - missing_after})")

        log.info("Attaching GSM IDs via GSM to CVCL for ACH bridge")
        cvcl_to_ach = (sample_info.dropna(subset=['RRID'])
                       .set_index('RRID')['DepMap_ID'].to_dict())
        geo = geo_info.dropna(subset=['Cellosaurus_ID']).copy() 
        geo['ACH_ID'] = geo['Cellosaurus_ID'].map(cvcl_to_ach)
        ach_to_gsm = (geo.dropna(subset=['ACH_ID'])
                      .groupby('ACH_ID')['Geo_accession']
                      .apply(list).to_dict())
        master_clean['GSM_IDs'] = master_clean['ACH_ID'].map(ach_to_gsm)
        log.info(f"GSM attachment: with GSM IDs={master_clean['GSM_IDs'].notna().sum()} "
                 f"| without={master_clean['GSM_IDs'].isna().sum()}")
        master_clean.rename(columns={'ACH_ID': 'ModelID'}, inplace=True)
        log.info(f"master_lookup final shape: {master_clean.shape}")
        result = save_csv(master_clean, 'master', 'master_lookup')
        if result is None or result != "successfull":
            raise Exception("Failed to save master_lookup.csv")
        log.info("Stage 1 (master_lookup) completed successfully")
        return master_clean
    except Exception as e:
        log.error(f"The pipeline failed while building master_lookup with error: {e}")
        raise

def build_gene_ensg_map():

    try:
        log.info("Building gene_ensg_map")

        log.info("Reading mart_export for protein-coding gene filter")
        mart = ut.data_reader('mart_export', 'mart_export')
        if mart is None or mart.empty:
            raise ValueError("mart_export is null or empty")
        protein_coding = set(
            mart[mart['Gene type'] == 'protein_coding']['Gene stable ID']
            .astype(str).str.strip()
        )
        log.info(f"Protein-coding ENSG IDs in mart: {len(protein_coding)}")
        depmap_path = os.path.join(base_path, 'data', 'raw', 'gene expression',
                                   '2_DepMap_OmicsExpressionAllGenesTPMLogp1Profile.csv')
        log.info(f"Reading File 2 header only (nrows=0) from: {depmap_path}")
        if not os.path.isfile(depmap_path):
            raise FileNotFoundError(f"File 2 not found at: {depmap_path}")
        header = pd.read_csv(depmap_path, nrows=0)
        log.info(f"File 2 column count: {len(header.columns)}")

        log.info("Parsing 'SYMBOL (ENSGID)' headers and filtering to protein-coding")
        symbol_to_ensg = {}
        missinggenename = []
        for col in header.columns:
            if '(' in col:
                symbol = col.split('(')[0].strip()
                ensg = col.split('(')[1].replace(')', '').strip().split('.')[0]
                if ensg in protein_coding:
                    symbol_to_ensg[symbol] = ensg
            else:
                if col in protein_coding:
                    missinggenename.append(col)

        gene_map = pd.DataFrame({
            'gene_symbol': list(symbol_to_ensg.keys()),
            'ensg_id': list(symbol_to_ensg.values()),
        })
        log.info(f"Primary symbol to ENSG mappings built: {len(gene_map)}")
        if missinggenename:
            metab_df = mart[(mart['Gene stable ID'].isin(missinggenename)) & (mart['Gene type'] == 'protein_coding')].copy()
            metab_df = metab_df.dropna(subset=['Gene name'])
            metab_df = metab_df.drop_duplicates(subset=['Gene stable ID'])
            metab_df = metab_df.rename(columns={'Gene name': 'gene_symbol', 'Gene stable ID':'ensg_id'})
            metab_df = metab_df[['gene_symbol', 'ensg_id']]
            if not metab_df.empty:
                gene_map_final = pd.concat([gene_map, metab_df], ignore_index=True)
            else:
                gene_map_final = gene_map
        else:
            gene_map_final = gene_map

        
        log.info(f"gene_ensg_map final rows: {len(gene_map_final)}")

        result = save_csv(gene_map_final, 'gene_maps', 'gene_ensg_map')
        if result is None or result != "successfull":
            raise Exception("Failed to save gene_ensg_map.csv")
        log.info("Stage 2 (gene_ensg_map) completed successfully")
        return gene_map_final
    except Exception as e:
        log.error(f"The pipeline failed while building gene_ensg_map with error: {e}")
        raise

def build_gene_protein_map(gene_ensg_map):
    try:
        log.info("Building gene_protein_map")
        prot_path = os.path.join(base_path, 'data', 'raw', 'gene expression',
                                 '4_Harmonized_MS_CCLE_Gygi_subsetted.csv')  # VERIFY path
        log.info(f"Reading File 4 header only (nrows=0) from: {prot_path}")
        if not os.path.isfile(prot_path):
            raise FileNotFoundError(f"File 4 not found at: {prot_path}")
        prot_header = pd.read_csv(prot_path, nrows=0)
        log.info(f"File 4 column count: {len(prot_header.columns)}")

        if gene_ensg_map is None or gene_ensg_map.empty:
            raise ValueError("gene_ensg_map is null or empty; cannot build protein map")
        symbol_to_ensg = gene_ensg_map.set_index('gene_symbol')['ensg_id'].to_dict()
        manual_alias = {
            'GBA1': 'GBA', 'CNK3/IPCEF1': 'IPCEF1', 'PHB1': 'PHB', 'MSL3B': 'MSL3',
            'YJU2B': 'YJU2', 'GATD3': 'GATD3A', 'NCF1B': 'NCF1', 'PMS2CL': 'PMS2',
            'EEF1A1P5': 'EEF1A1', 'EP400P1': 'EP400', 'RPS10P5': 'RPS10',
        }

        log.info("Parsing 'UNIPROT (GENE)' headers and resolving each to ENSG")
        rows = []
        alias_hits = 0
        for col in prot_header.columns:
            if '(' in col:
                protein_id = col.split('(')[0].strip()
                m = re.search(r'\((.+)\)', col)
                gene_symbol = m.group(1).strip() if m else col
            else:
                protein_id = col
                gene_symbol = col
            ensg = symbol_to_ensg.get(gene_symbol)
            if ensg is None and gene_symbol in manual_alias:
                ensg = symbol_to_ensg.get(manual_alias[gene_symbol])
                if ensg is not None:
                    alias_hits += 1
            rows.append({'protein_id': protein_id,
                         'gene_symbol': gene_symbol,
                         'ensg_id': ensg})

        gene_protein_map = pd.DataFrame(rows)[['gene_symbol', 'ensg_id', 'protein_id']]
        log.info(f"gene_protein_map rows: {len(gene_protein_map)} "
                 f"| resolved via alias table: {alias_hits}")
        log.info(f"Proteins with ENSG: {gene_protein_map['ensg_id'].notna().sum()} "
                 f"| without ENSG: {gene_protein_map['ensg_id'].isna().sum()}")

        result = save_csv(gene_protein_map, 'gene_maps', 'gene_protein_map')
        if result is None or result != "successfull":
            raise Exception("Failed to save gene_protein_map.csv")
        log.info("Stage 3 (gene_protein_map) completed successfully")
        return gene_protein_map
    except Exception as e:
        log.error(f"The pipeline failed while building gene_protein_map with error: {e}")
        raise

def build_lookup_tables():
    try:
        log.info("Starting lookup-table construction")
        master_lookup = build_master_lookup()
        gene_ensg_map = build_gene_ensg_map()
        gene_protein_map = build_gene_protein_map(gene_ensg_map)
        log.info("All three lookup artifacts built and saved successfully")
        return master_lookup, gene_ensg_map, gene_protein_map
    except Exception as e:
        log.error(f"The pipeline failed in the lookup-table construction with error: {e}")
        raise
