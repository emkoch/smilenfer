import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import smilenfer.statistics as sstats
import smilenfer.simulation as sim

# ---- data ----
BASE_DIR = "../data"
SFS_PILE = os.path.join(BASE_DIR, "SFS_pile", "tenn_eur_pile.pkl")
sfs_pile_eur = sim.truncate_pile(pickle.load(open(SFS_PILE, "rb")), 1e-8)

# ---- grids ----
MIN_X = 0.01
global_x_ud,  SS_ud,  tau_ud  = sstats.build_simple_grid(sfs_pile_eur, min_x=0.01, n_points=1000)
global_x_std, SS_std, tau_std = sstats.build_integration_grid_s(sfs_pile_eur, min_x=0.01, n_points=1000)

global_x_ud_Ip,  _, _, S_p_ud,  int_grid_ud  = sstats.build_integration_grid(sfs_pile_eur, min_x=MIN_X, n_points=1000, ud=True)
global_x_std_Ip, _, _, S_p_std, int_grid_std = sstats.build_integration_grid(sfs_pile_eur, min_x=MIN_X, n_points=1000, ud=False)

# ---- helpers ----
def get_sfs_s(SS, SS_set, tau):
    idx = np.searchsorted(SS_set, SS)
    return tau[:, idx]

def normalize_sfs(xx, sfs):
    return sfs / np.trapz(sfs, xx)

# ---- shared styling ----
cmap   = plt.get_cmap("viridis")
colors = [cmap(v) for v in np.linspace(0, 1, 4)]

def sfs_curve(ax, xs, S_vals, SS_set, tau, *, ls="-"):
    for col, S in zip(colors, S_vals):
        yy = get_sfs_s(S, SS_set, tau) + np.flip(get_sfs_s(S, SS_set, tau))
        ax.plot(xs, normalize_sfs(xs, yy), color=col, linestyle=ls, label=f"S={S}")

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

# Panel A – Underdominance vs Standard (1-T stabilizing)
ax = axes[0]
sfs_curve(ax, global_x_ud,  [0, 2, 20, 200], SS_ud,  tau_ud)            # solid
sfs_curve(ax, global_x_std, [0, 1, 10, 100], SS_std, tau_std, ls="--")  # dashed
ax.set_title("1-T stabilizing")
ax.set_yscale("log")
ax.set_xscale("logit")
ax.set_ylabel("SFS")

# Panel B – Pleiotropic stabilizing (integration variant)
ax = axes[1]
sfs_curve(ax, global_x_ud_Ip,  [0, 20, 200, 2000], S_p_ud,  int_grid_ud.T)            # solid
sfs_curve(ax, global_x_std_Ip, [0, 10, 100, 1000], S_p_std, int_grid_std.T, ls="--")  # dashed
ax.set_title("Pleiotropic stabilizing")
ax.set_yscale("log")
ax.set_xscale("logit")

for ax in axes:
    ax.set_xlabel("Allele frequency")

# Legends: color = selection strength, line style = model
handles_colour = [Line2D([0], [0], color=c, lw=2) for c in colors]
axes[0].legend(handles_colour, ["S=0", "S=2", "S=20", "S=200"], title="Selection (S)", loc="lower right")

handles_colour = [Line2D([0], [0], color=c, lw=2) for c in colors]
leg2 = axes[1].legend(handles_colour, ["Sp=0", "Sp=2", "Sp=20", "Sp=200"], title="Selection (Sp)", loc="lower right")
axes[1].add_artist(leg2)

handles_style = [Line2D([0], [0], color='k', lw=2, ls=ls) for ls in ['-', '--']]
axes[1].legend(handles_style, ["Underdominance", "Standard"], title="Model", loc="lower left")

plt.tight_layout()
fig.savefig("ud_v_std_example_sfs.pdf", bbox_inches="tight", dpi=300)
