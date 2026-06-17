import pandas as pd
import os
import numpy as np

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

fl_pos_GC = "path_to_positive_GC_featurelist.csv"
fl_neg_GC = "path_to_negative_GC_featurelist.csv"
fl_pos_BDD = "path_to_positive_BDD_featurelist.csv"
fl_neg_BDD = "path_to_negative_BDD_featurelist.csv"
qc_list = "path_to_QC_names.csv"
literature_tp_database = "path_to_literature_TP_database.csv"

save_simple_positives     = False
save_replicate_positives  = False
save_literature_comparison = False
output_dir = "output_directory"

n_reps = 5

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

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


def replicate_filter(df: pd.DataFrame, require_reps: int,
                     save_rp: bool = False, name: str = "") -> pd.DataFrame:
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


def both_modes_literature_comparison(pos_df: pd.DataFrame, neg_df: pd.DataFrame,
                                      literature_path: str, electrode: str) -> pd.DataFrame:
    fl = pd.concat([pos_df, neg_df], ignore_index=True)
    fl["clean_name"] = fl["compound"].str.replace(r"[a-z]$", "", regex=True)

    exp_tps = pd.read_csv(literature_path)
    exp_tps = exp_tps[exp_tps["Mode"].notna()].copy()

    def match_row(row):
        pos_match = not fl[(fl["clean_name"] == row["pos"]) & (fl["formula"] == row["Formula"])].empty
        neg_match = not fl[(fl["clean_name"] == row["neg"]) & (fl["formula"] == row["Formula"])].empty
        if not pos_match and not neg_match:
            return None
        return {
            "pos":              row["pos"] if pos_match else np.nan,
            "neg":              row["neg"] if neg_match else np.nan,
            "formula":          row["Formula"],
            "mode":             row["Mode"],
            "overall reaction": row["Overall_Reaction"],
            "electrode":        electrode
        }

    results = [r for row in exp_tps.itertuples(index=False)
               if (r := match_row(row._asdict())) is not None]

    total, matched = len(exp_tps), len(results)
    print(f"Matched {matched}/{total} expected ETPs "
          f"({matched/total*100:.1f}%) for {electrode}")

    return pd.DataFrame(results)


def merge_bdd_gc(bdd_df: pd.DataFrame, gc_df: pd.DataFrame,
                  save_comp: bool = False) -> None:
    merged = (
        pd.concat([bdd_df, gc_df], ignore_index=True)
        .groupby(["formula", "mode", "overall reaction"], dropna=False)
        .agg({
            "pos":      lambda x: x.dropna().unique()[0] if x.dropna().size > 0 else np.nan,
            "neg":      lambda x: x.dropna().unique()[0] if x.dropna().size > 0 else np.nan,
            "electrode": lambda x: "/".join(sorted(set(x)))
        })
        .reset_index()
        [["pos", "neg", "formula", "mode", "overall reaction", "electrode"]]
    )

    print(f"{len(merged)} total unique ETPs in combined comparison")

    if save_comp:
        merged.to_csv(os.path.join(output_dir, "literature_comparison_combined.csv"), index=False)

# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    print("Processing feature lists...\n")

    kwargs = dict(qc_list_path=qc_list, require_reps=n_reps,
                  save_sp=save_simple_positives, save_rp=save_replicate_positives)

    results = {
        "GC_pos":  simple_replicate_positives(fl_pos_GC,  **kwargs),
        "GC_neg":  simple_replicate_positives(fl_neg_GC,  **kwargs),
        "BDD_pos": simple_replicate_positives(fl_pos_BDD, **kwargs),
        "BDD_neg": simple_replicate_positives(fl_neg_BDD, **kwargs),
    }

    print("\nFeature lists processed, starting literature comparison...\n")

    lit_GC  = both_modes_literature_comparison(results["GC_pos"],  results["GC_neg"],
                                                literaure_tp_database, "GC")
    lit_BDD = both_modes_literature_comparison(results["BDD_pos"], results["BDD_neg"],
                                                literaure_tp_database, "BDD")
    merge_bdd_gc(lit_BDD, lit_GC, save_comp=save_literature_comparison)

    print("\nComparison completed.")

if __name__ == "__main__":
    main()