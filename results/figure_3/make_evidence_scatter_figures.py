import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import smilenfer.plotting as splot
import smilenfer.posterior as spost

splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

FIT_FILE = "../first_mode_fits/original_traits/opt_results_original_traits_eur_post.csv"
TRAIT_NAMES = spost.original_trait_names


def trait_label(trait):
    return TRAIT_NAMES.get(trait, trait.replace("_", " ").title())

fit_df = pd.read_csv(FIT_FILE)

ll_plei_main = fit_df.ll_plei.to_numpy() - fit_df.ll_neut.to_numpy()
ll_plei_stab_main = fit_df.ll_plei.to_numpy() - fit_df.ll_stab.to_numpy()

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.scatter(2 * ll_plei_main, 2 * ll_plei_stab_main)

for ii, trait in enumerate(fit_df.trait.to_numpy()):
    ax.annotate(
        trait_label(trait),
        (2 * ll_plei_main[ii], 2 * ll_plei_stab_main[ii]),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.2),
    )

ax.set_xlabel(
    r"Evidence for selection $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{neutral}})$",
    fontweight="bold",
)
ax.set_xscale("symlog", linthresh=10)
ax.set_xlim([-2, 2000])
ax.set_xticks([0, 1, 10, 100, 1000])
ax.set_xticklabels([0, 1, 10, 100, 1000])

ax.set_ylabel(
    r"Evidence for pleiotropy $(-\Delta \mathrm{AIC}_{\mathrm{PLEI} - \mathrm{1T}})$",
    fontweight="bold",
)
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim([-4, 2000])
ax.set_yticks([0, 1, 10, 100])
ax.set_yticklabels([0, 1, 10, 100])

fig.tight_layout()
fig.savefig("pleiotropy_evidence.pdf", bbox_inches="tight")

ll_dir_main = fit_df.ll_dir.to_numpy() - fit_df.ll_neut.to_numpy()
i1_ests_main = fit_df.I1_dir.to_numpy()

np.random.seed(1)
x_jitter = np.random.uniform(-1e-1, 1e-1, len(fit_df))
y_jitter = np.random.uniform(-1e-5, 1e-5, len(fit_df))

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.scatter(2 * ll_dir_main, i1_ests_main)

for ii, trait in enumerate(fit_df.trait.to_numpy()):
    ax.annotate(
        trait_label(trait),
        (2 * ll_dir_main[ii] + x_jitter[ii], i1_ests_main[ii] + y_jitter[ii]),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.2),
    )

ax.set_xlabel(
    r"Evidence for selection $(-\Delta \mathrm{AIC}_{\mathrm{DIR} - \mathrm{neutral}})$",
    fontweight="bold",
)
ax.set_xscale("symlog", linthresh=1)
ax.set_ylabel(r"$I_1$", fontweight="bold")
ax.set_yscale("symlog", linthresh=1e-4)

fig.tight_layout()
fig.savefig("I_1_directionality.pdf", bbox_inches="tight")
