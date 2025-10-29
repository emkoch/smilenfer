"""
Single Trait smilenfer Analysis

This script performs single-trait polygenic selection inference.
It analyzes GWAS summary statistics to infer selection parameters.
These summary statistics should be LD clumped so that the input
variants are approximately independent.

Usage:
    python single_trait_smilenfer.py --input <input_file> [options]

Required input file columns:
    - raf: Risk / trait-increasing allele frequency
    - rbeta: Effect size magnitude of risk / trait-increasing allele (beta coefficient)
    - se: Standard error
    - pvalue OR neglog10p: P-values (either format accepted)

Options:
    --input: Input data file (required)
    --output: Output file for results (default: output.csv)
    --pvalue_threshold: P-value threshold for significance (default: 5e-8)
    --true_rbeta_col: Column name for "true" beta values. These will be treated as true effect sizes while rbeta is used for ascertainment (optional)
    --sfs_pile: SFS pile option - 'eur' or 'jpt' (default: eur)
    --explicit_pvalue_filter: Apply explicit p-value filtering

Example:
    python single_trait_smilenfer.py --input results/data/final/original_traits/processed.asthma.snps_low_r2.tsv --output asthma_results.csv --pvalue_threshold 5e-8
"""

import argparse
import pickle
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import smilenfer.statistics as sstats
import smilenfer.simulation as sim

def check_data(data):
    required_cols = ['raf', 'rbeta', 'se']
    pvalue_cols = ['pvalue', 'neglog10p']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    if not any(col in data.columns for col in pvalue_cols):
        raise ValueError("Missing required p-value column: either 'pvalue' or 'neglog10p' must be present")
    if 'pvalue' in data.columns:
        data['neglog10p'] = -np.log10(data['pvalue'])

    return data

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description = "Single Trait smilenfer Analysis")
    parser.add_argument("--input", type=str, required=True, help="Input data file")
    parser.add_argument("--output", type=str, default="output.csv", help="Output file for results")
    parser.add_argument("--pvalue_threshold", type=float, default=5e-8, help="P-value threshold for significance")
    parser.add_argument("--true_rbeta_col", type=str, default=None, help="Column name for beta values")
    parser.add_argument("--sfs_pile", type=str, choices=['eur', 'jpt'], default='eur', help="SFS pile option: 'eur' or 'jpt' (default: eur)")
    parser.add_argument("--explicit_pvalue_filter", action="store_true", help="Whether to explicitly filter by p-value threshold")
    return parser.parse_args(argv)

def print_usage():
    """Print usage information when no arguments are provided."""
    print(__doc__)
    print("\nError: No input file specified. Use --input to specify the input data file.")
    print("Run with --help for detailed argument information.")

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv or (len(argv) == 1 and argv[0] in ['-h', '--help']):
        if not argv:
            print_usage()
            sys.exit(1)
    
    args = parse_arguments(argv)
    data = pd.read_csv(args.input, sep=None, engine="python", compression="infer")
    data = check_data(data)

    data["n_eff"] = 1 / (
            2
            * data["se"] ** 2
            * data["raf"]
            * (1 - data["raf"])
        )
    
    n_eff_median = np.nanmedian(data["n_eff"])

    v_exp = np.array(2 * data["rbeta"] ** 2 * data["raf"] * (1 - data["raf"]))
    v_cut = stats.chi2.isf(args.pvalue_threshold, df=1) / n_eff_median

    raf = data["raf"].to_numpy()
    rbeta = data["rbeta"].to_numpy()

    if args.true_rbeta_col and args.true_rbeta_col in data.columns:
        rbeta_post = data[args.true_rbeta_col].to_numpy()
    elif args.true_rbeta_col is not None:
        raise ValueError(f"Specified true_rbeta_col '{args.true_rbeta_col}' not found in data columns")
    else:
        rbeta_post = None

    keep = v_exp > v_cut

    if args.explicit_pvalue_filter:
        pvalues = 10 ** (-data["neglog10p"].to_numpy())
        keep = keep & (pvalues < args.pvalue_threshold)

    raf_keep = raf[keep]
    rbeta_keep = rbeta[keep]
    rbeta_post_keep = rbeta_post[keep] if rbeta_post is not None else None

    script_dir = os.path.dirname(os.path.realpath(__file__))

    pile_map = {
        "jpt": os.path.join(script_dir, "results", "data", "SFS_pile", "joug_jpt_pile.pkl"),
        "eur": os.path.join(script_dir, "results", "data", "SFS_pile", "tenn_eur_pile.pkl"),
    }

    if args.sfs_pile not in pile_map:
        raise ValueError("other sfs_pile options not implemented")

    sfs_pile_loc = pile_map[args.sfs_pile]
    
    with open(sfs_pile_loc, "rb") as f:
        sfs_pile = sim.truncate_pile(pickle.load(f), 1e-8)

    opt_result = sstats.infer_all_standard(
        sfs_pile,
        10000,
        raf_keep,
        rbeta_keep,
        v_cut,
        min_x=0.01,
        n_points=1000,
        n_x=1000,
        beta_obs=rbeta_post_keep,
    )

    def _as_float(x):
        try:
            return float(x)
        except Exception:
            return float(x[0])

    rows = []

    rows.append(("neut", "-", float(opt_result["ll_neut"]), "-", "-"))

    r = opt_result["I1_effects"]
    I1 = _as_float(r.x)
    rows.append(("I1", f"I1={I1:.6g}", -float(r.fun), bool(r.success), getattr(r, "nit", "-")))

    r = opt_result["I2_effects"]
    I2 = 10.0 ** _as_float(r.x)
    rows.append(("I2", f"I2={I2:.6g}", -float(r.fun), bool(r.success), getattr(r, "nit", "-")))

    r = opt_result["Ip_effects"]
    Ip = 10.0 ** _as_float(r.x)
    rows.append(("Ip", f"Ip={Ip:.6g}", -float(r.fun), bool(r.success), getattr(r, "nit", "-")))

    r = opt_result["full_effects"]
    I1_full = float(r.x[0])
    I2_full = 10.0 ** float(r.x[1])
    rows.append(("full", f"I1={I1_full:.6g}, I2={I2_full:.6g}",
                -float(r.fun), bool(r.success), getattr(r, "nit", "-")))

    w_model, w_params, w_ll, w_success, w_nit = 6, 42, 12, 8, 6

    print(f'{"model":<{w_model}} {"params":<{w_params}} {"loglik":>{w_ll}} {"success":>{w_success}} {"nit":>{w_nit}}')
    for name, params, ll, ok, nit in rows:
        print(f'{name:<{w_model}} {params:<{w_params}} {ll:>{w_ll}.6g} {str(ok):>{w_success}} {str(nit):>{w_nit}}')

    results_df = pd.DataFrame(rows, columns=["model", "params", "loglik", "success", "nit"])
    results_df = results_df.drop(columns=["success", "nit"])
    results_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()