import pandas as pd
import os

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

fl_pos_GC_ref = "path_to_positive_GC_reference_featurelist.csv"
fl_neg_GC_ref = "path_to_negative_GC_reference_featurelist.csv"
fl_pos_GC_lib = "path_to_positive_GC_library_featurelist.csv"
fl_neg_GC_lib = "path_to_negative_GC_library_featurelist.csv"
fl_pos_BDD_ref = "path_to_positive_BDD_reference_featurelist.csv"
fl_neg_BDD_ref = "path_to_negative_BDD_reference_featurelist.csv"
fl_pos_BDD_lib = "path_to_positive_BDD_library_featurelist.csv"
fl_neg_BDD_lib = "path_to_negative_BDD_library_featurelist.csv"
qc_list = "path_to_QC_names.csv"

save_simple_positives        = False
save_replicate_positives     = False
save_intersequence_positives = False
output_dir = "output_directory"

n_reps_ref = 5
n_reps_lib = 3

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def load_qc_names(path: str) -> list[str]:
    return pd.read_csv(path).iloc[:, 0].dropna().astype(str).str.strip().str.lower().tolist()


def process_featurelist(path: str, qc_names: list[str], save_sp: bool = False) -> pd.DataFrame:
    """Load, filter, and melt one feature list into long format."""
    fl_name = os.path.basename(path)
    df = pd.read_csv(path, low_memory=False)

    df = df[df["formulas:formulas"].notna()]
    df = df[df["fragment_scans"] != 0]
    df = df[df["compound_db_identity:compound_name"]
            .str.lower().str.strip().fillna("")
            .apply(lambda c: any(q in c for q in qc_names))]

    name_cols = ["compound_db_identity:compound_name", "formulas:formulas"]
    frag_cols = [c for c in df.columns if ":fragment_scans" in c]
    df = df[[c for c in name_cols if c in df.columns] + frag_cols]

    if save_sp:
        df.to_csv(os.path.join(output_dir, f"simple_positives_{fl_name}.csv"), index=False)

    qc_cols = [c for c in df.columns if "_QC_" in c]
    melted = df.melt(id_vars=name_cols, value_vars=qc_cols,
                     var_name="measurement", value_name="fragment_scans")

    pattern = r".*_(?P<potential>-?\d+)mV_\d+(?P<suffix>[a-z]?)\.mzML:fragment_scans$"
    meta = melted["measurement"].str.extract(pattern)
    meta["potential"] = meta.apply(
        lambda r: f"{r['potential']}_{r['suffix']}" if r["suffix"] else r["potential"], axis=1
    )

    melted = melted.rename(columns={
        "compound_db_identity:compound_name": "compound",
        "formulas:formulas": "formula"
    })
    melted["potential"] = meta["potential"].values
    return melted[["compound", "formula", "potential", "fragment_scans"]]


def replicate_filter(df: pd.DataFrame, require_reps: int, save_rp: bool = False, name: str = "") -> pd.DataFrame:
    """Keep compounds with >= require_reps MS² scans in at least one potential."""
    def enough_reps(g):
        return (g["fragment_scans"].fillna(0).astype(float) >= 1).sum() >= require_reps

    status = (df.groupby(["compound", "formula", "potential"])
                .apply(enough_reps)
                .reset_index(name="ok"))
    keep = status.groupby("compound")["ok"].transform("any")
    result = status[keep][["compound", "formula"]].drop_duplicates().reset_index(drop=True)

    if save_rp and name:
        result.to_csv(os.path.join(output_dir, f"replicate_positives_{name}.csv"), index=False)

    return result


def clean_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Strip EC potential suffix from compound names and remove duplicates."""
    df = df.copy()
    df["clean_name"] = df["compound"].str.replace(r"_[^_]+$", "", regex=True)
    return df.drop_duplicates(subset=["formula", "clean_name"]).reset_index(drop=True)

# ─── MAIN FUNCTIONS ──────────────────────────────────────────────────────────

def simple_replicate_positives(path: str, qc_list_path: str,
                                require_reps: int = 3,
                                save_sp: bool = False, save_rp: bool = False) -> pd.DataFrame:
    fl_name  = os.path.basename(path)
    qc_names = load_qc_names(qc_list_path)

    melted = process_featurelist(path, qc_names, save_sp=save_sp)
    result = replicate_filter(melted, require_reps, save_rp=save_rp, name=fl_name)

    print(f"{fl_name}: Done")
    return result


def intersequence_reproducibility(ref_pos: pd.DataFrame, ref_neg: pd.DataFrame,
                                   lib_pos: pd.DataFrame, lib_neg: pd.DataFrame) -> None:
    ref = clean_and_dedup(pd.concat([ref_pos, ref_neg], ignore_index=True))
    lib = clean_and_dedup(pd.concat([lib_pos, lib_neg], ignore_index=True))

    ref_set = set(zip(ref["clean_name"], ref["formula"]))
    lib_set = set(zip(lib["clean_name"], lib["formula"]))

    union   = ref_set | lib_set
    overlap = ref_set & lib_set
    perc    = len(overlap) / len(union) * 100 if union else 0

    print(f"Total unique ETP features across sequences and ionisation modes: {len(union)}")
    print(f"Exclusive in reference sequence: {len(ref_set - lib_set)}")
    print(f"Exclusive in library sequence:   {len(lib_set - ref_set)}")
    print(f"Overlap: {len(overlap)}")
    print(f"Intersequence reproducibility: {perc:.1f}%\n")

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    print("Processing feature lists...\n")

    kwargs_ref = dict(qc_list_path=qc_list, require_reps=n_reps_ref,
                      save_sp=save_simple_positives, save_rp=save_replicate_positives)
    kwargs_lib = dict(qc_list_path=qc_list, require_reps=n_reps_lib,
                      save_sp=save_simple_positives, save_rp=save_replicate_positives)

    results = {
        "GC_ref_pos":  simple_replicate_positives(fl_pos_GC_ref,  **kwargs_ref),
        "GC_ref_neg":  simple_replicate_positives(fl_neg_GC_ref,  **kwargs_ref),
        "GC_lib_pos":  simple_replicate_positives(fl_pos_GC_lib,  **kwargs_lib),
        "GC_lib_neg":  simple_replicate_positives(fl_neg_GC_lib,  **kwargs_lib),
        "BDD_ref_pos": simple_replicate_positives(fl_pos_BDD_ref, **kwargs_ref),
        "BDD_ref_neg": simple_replicate_positives(fl_neg_BDD_ref, **kwargs_ref),
        "BDD_lib_pos": simple_replicate_positives(fl_pos_BDD_lib, **kwargs_lib),
        "BDD_lib_neg": simple_replicate_positives(fl_neg_BDD_lib, **kwargs_lib),
    }

    print("\nStarting intersequence reproducibility calculation...\n")

    print("GC:")
    intersequence_reproducibility(results["GC_ref_pos"],  results["GC_ref_neg"],
                                   results["GC_lib_pos"],  results["GC_lib_neg"])
    print("BDD:")
    intersequence_reproducibility(results["BDD_ref_pos"], results["BDD_ref_neg"],
                                   results["BDD_lib_pos"], results["BDD_lib_neg"])

    print("Calculation completed.")

if __name__ == "__main__":
    main()