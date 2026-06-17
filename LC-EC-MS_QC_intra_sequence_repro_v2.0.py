import pandas as pd
import os

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

fl_pos_GC = "path_to_positive_GC_featurelist.csv"
fl_neg_GC = "path_to_negative_GC_featurelist.csv"
fl_pos_BDD = "path_to_positive_BDD_featurelist.csv"
fl_neg_BDD = "path_to_negative_BDD_featurelist.csv"
qc_list = "path_to_QC_names.csv"

save_simple_positives    = False
save_replicate_positives = False
output_dir = "output_directory"
n_reps = 5

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def load_qc_names(path: str) -> list[str]:
    return pd.read_csv(path).iloc[:, 0].dropna().astype(str).str.strip().str.lower().tolist()


def process_featurelist(path: str, qc_names: list[str]) -> pd.DataFrame:
    # Load, filter, and melt one feature list into long format.
    df = pd.read_csv(path, low_memory=False)

    df = df[df["formulas:formulas"].notna()]
    df = df[df["fragment_scans"] != 0]
    df = df[df["compound_db_identity:compound_name"]
            .str.lower().str.strip().fillna("")
            .apply(lambda c: any(q in c for q in qc_names))]

    name_cols = ["compound_db_identity:compound_name", "formulas:formulas"]
    frag_cols = [c for c in df.columns if ":fragment_scans" in c]
    df = df[[c for c in name_cols if c in df.columns] + frag_cols]

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


def replicate_filter(df: pd.DataFrame, require_reps: int) -> pd.DataFrame:
    # Keep compounds with >= require_reps MS² scans in at least one potential."""
    def enough_reps(g):
        return (g["fragment_scans"].fillna(0).astype(float) >= 1).sum() >= require_reps

    status = (df.groupby(["compound", "formula", "potential"])
                .apply(enough_reps)
                .reset_index(name="ok"))
    keep = status.groupby("compound")["ok"].transform("any")
    return status[keep][["compound", "formula"]].drop_duplicates().reset_index(drop=True)


# ─── MAIN FUNCTION ───────────────────────────────────────────────────────────

def simple_replicate_positives(path_pos: str, path_neg: str, qc_list_path: str,
                                require_reps: int = 3,
                                save_sp: bool = False, save_rp: bool = False):
    qc_names = load_qc_names(qc_list_path)
    name_pos, name_neg = os.path.basename(path_pos), os.path.basename(path_neg)

    df_pos = process_featurelist(path_pos, qc_names)
    df_neg = process_featurelist(path_neg, qc_names)

    # Simple positives
    combined = pd.concat([df_pos, df_neg], ignore_index=True)
    combined["clean_name"] = combined["compound"].str.replace(r"_[^_]+$", "", regex=True)
    simple_pos = combined.drop_duplicates(subset=["formula", "clean_name"])

    if save_sp:
        for name, df in [(name_pos, df_pos), (name_neg, df_neg)]:
            df.to_csv(os.path.join(output_dir, f"simple_positives_{name}.csv"), index=False)

    # Replicate positives
    rp = pd.concat([replicate_filter(df_pos, require_reps),
                    replicate_filter(df_neg, require_reps)], ignore_index=True)
    rp["clean_name"] = rp["compound"].str.replace(r"_[^_]+$", "", regex=True)
    rp = rp.drop_duplicates(subset=["formula", "clean_name"]).reset_index(drop=True)

    before, after = len(simple_pos), len(rp)
    perc = after / before * 100 if before > 0 else 0
    print(f"{name_pos} and {name_neg} combined:")
    print(f"{before} simple-positive features")
    print(f"{after} replicate-positive features")
    print(f"Intrasequence reproducibility: {perc:.1f}%\n")

    if save_rp:
        rp.to_csv(os.path.join(output_dir, "replicate_positives.csv"), index=False)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    print("Calculating intrasequence reproducibility...\n")
    simple_replicate_positives(fl_pos_GC,  fl_neg_GC,  qc_list, require_reps=n_reps,
                                save_sp=save_simple_positives, save_rp=save_replicate_positives)
    simple_replicate_positives(fl_pos_BDD, fl_neg_BDD, qc_list, require_reps=n_reps,
                                save_sp=save_simple_positives, save_rp=save_replicate_positives)
    print("Calculation completed.")

if __name__ == "__main__":
    main()