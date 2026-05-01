from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ORIGINAL_DIR = SCRIPT_DIR.parent / "data" / "final" / "original_traits"
ORIGINAL_FITS = (
    SCRIPT_DIR.parent
    / "all_opt_fits"
    / "original_traits"
    / "opt_results_original_traits_eur_post.csv"
)
SFS_PILE = SCRIPT_DIR.parent / "data" / "SFS_pile" / "tenn_eur_pile.pkl"

P_THRESH = 5e-8
MIN_X = 0.01
NE = 10_000

TRAIT_LABELS = {
    "bmi": "BMI",
    "cad": "CAD",
    "dbp": "DBP",
    "hdl": "HDL",
    "height": "Height",
    "ldl": "LDL",
    "rbc": "RBC",
    "sbp": "SBP",
    "scz": "SCZ",
    "t2d": "T2D",
    "triglycerides": "Triglycerides",
    "wbc": "WBC",
}

FIT_ORDER = ["full_original", "matched_original_post", "raw_all", "raw_z2", "raw_z5"]
FIT_LABELS = {
    "full_original": "Full original",
    "matched_original_post": "Matched original",
    "raw_all": "BBJ replace all",
    "raw_z2": r"BBJ replace $Z_{\mathrm{exp}}\geq2$",
    "raw_z5": r"BBJ replace $Z_{\mathrm{exp}}\geq5$",
}
FIT_COLORS = {
    "full_original": "#222222",
    "matched_original_post": "#FE6100",
    "raw_all": "#785EF0",
    "raw_z2": "#648FFF",
    "raw_z5": "#DC267F",
}
FIT_OFFSETS = {
    "full_original": -0.32,
    "matched_original_post": -0.16,
    "raw_all": 0.0,
    "raw_z2": 0.16,
    "raw_z5": 0.32,
}


def trait_order(traits):
    return sorted(traits, key=lambda x: TRAIT_LABELS.get(x, x))


def load_original_fit_set(trait):
    path = ORIGINAL_DIR / f"processed.{trait}.snps_low_r2.tsv"
    if not path.exists():
        return None

    df = pd.read_csv(path, sep="\t")
    n_eff_med = np.nanmedian(df["n_eff"])
    v_cut = stats.chi2.isf(P_THRESH, df=1) / n_eff_med
    v_exp = 2 * df["raf"] * (1 - df["raf"]) * df["rbeta"] ** 2
    keep = v_exp > v_cut

    out = df.loc[keep, ["snp", "raf", "rbeta", "PosteriorMean"]].copy()
    out["maf_orig"] = np.minimum(out["raf"].to_numpy(), 1 - out["raf"].to_numpy())
    out["v_cut"] = v_cut
    out["trait"] = trait
    return out


def slope_through_origin(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return np.nan
    denom = np.sum(x[ok] ** 2)
    if denom == 0:
        return np.nan
    return np.sum(x[ok] * y[ok]) / denom
