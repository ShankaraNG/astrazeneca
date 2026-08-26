import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
fusion_path = os.path.join(DATA, "scoring", "gene properties", "5_OmicsFusionFilteredSupplementary.csv")
map_path    = os.path.join(DATA, "lookup", "gene_maps", "gene_ensg_map.csv")

f = pd.read_csv(fusion_path)
print(f"fusion file rows: {len(f)}")
gmap = pd.read_csv(map_path)
f = f.dropna(subset=['gene1_ENSG_ID', 'gene2_ENSG_ID'])
# f = f[f['ModelConditionID'] == 'Yes']
print(f"rows with both partner ENSG ids present: {len(f)}")

pair_counts = (f.groupby(['gene1_ENSG_ID', 'gene2_ENSG_ID'])['ModelID']
                 .nunique()
                 .sort_values(ascending=False)
                 .reset_index()
                 .rename(columns={'ModelID': 'n_cell_lines'}))

filtered_counts = pair_counts[pair_counts['n_cell_lines'] > 10]

valid_ensgs = set(gmap['ensg_id'])
filtered_pairs = filtered_counts[
    filtered_counts['gene1_ENSG_ID'].isin(valid_ensgs) & 
    filtered_counts['gene2_ENSG_ID'].isin(valid_ensgs)
]

print(filtered_pairs.to_string(index=False))

gmap = pd.read_csv(map_path)
ensg_to_symbol = dict(zip(gmap['ensg_id'], gmap['gene_symbol']))

print("\nReady-to-test gene lists (top pairs where BOTH map to a symbol)")
shown = 0
for _, row in pair_counts.iterrows():
    if shown >= 5:
        break
    g1, g2, n = row['gene1_ENSG_ID'], row['gene2_ENSG_ID'], row['n_cell_lines']
    s1 = ensg_to_symbol.get(g1)
    s2 = ensg_to_symbol.get(g2)
    if not s1 or not s2:
        continue
    print(f"\n  fusion {g1} + {g2}  (in {n} cell line(s))")
    print(f"  targetgenelist = ['{s1}', '{s2}']")
    shown += 1
if shown == 0:
    print("  (no top pair had BOTH partners in gene_ensg_map — widen the search or check the map)")

print("\nformat sanity check (must look identical)")
if len(f):
    print("example fusion gene1_ENSG_ID:", f['gene1_ENSG_ID'].iloc[0])
print("example map ensg_id         :", gmap['ensg_id'].iloc[0])
print("(both should be bare ENSG, no version suffix like .12)")