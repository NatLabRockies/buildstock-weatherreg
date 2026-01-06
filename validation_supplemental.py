import csv


def get_sets(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        src = set()
        sim = set()
        pairs = set()
        for row in reader:
            a = row["in.nhgis_county_gisjoin"]
            b = row["in.as_simulated_nhgis_county_gisjoin"]
            src.add(a)
            sim.add(b)
            pairs.add((a, b))
    return src, sim, pairs


ref_src, ref_sim, ref_pairs = get_sets(
    "/projects/geohc/geo_predict/outputs/outputs_2025-12-10-15-53-30/com_meta_master_upgrade0.csv"
)
reg_src, reg_sim, reg_pairs = get_sets(
    "/projects/geohc/geo_predict/outputs/outputs_2025-12-02-09-29-48/com_meta_master_upgrade0.csv"
)

print("src counties: ref", len(ref_src), "reg", len(reg_src), "common", len(ref_src & reg_src))
print("sim counties: ref", len(ref_sim), "reg", len(reg_sim), "common", len(ref_sim & reg_sim))
print("pair count: ref", len(ref_pairs), "reg", len(reg_pairs), "common", len(ref_pairs & reg_pairs))
print("pairs only ref", len(ref_pairs - reg_pairs), "only reg", len(reg_pairs - ref_pairs))

ref_only = list(ref_pairs - reg_pairs)[:5]
reg_only = list(reg_pairs - ref_pairs)[:5]
print("sample pair only ref", ref_only)
print("sample pair only reg", reg_only)
