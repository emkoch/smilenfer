import smilenfer.dfe as dfe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import smilenfer.plotting as splot

matplotlib.rcParams.update({'font.size': 18})
splot._plot_params()

s_set = np.linspace(-6, -1.5, 1000)
simons_f = dfe.simons_ssd(s_set)
simons_F = dfe.simons_ssd_F(s_set)
simons_F_inv = dfe.simons_ssd_F_inv(np.linspace(0, 1, 1000))
s_sample = dfe.simons_ssd_sample(10000)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(s_set, simons_f, label="Simons et al. (2022)")
ax.set_xlabel(r"$\log_{10}(s_{ud})$")
ax.set_ylabel(r"$f(\log_{10}(s_{ud}))$")
fig.savefig("simons_dfe_f.pdf", bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(s_set, simons_F, label="Simons et al. (2022)")
ax.set_xlabel(r"$\log_{10}(s)$")
ax.set_ylabel(r"$F(s)$")
fig.savefig("simons_dfe_F.pdf", bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(np.linspace(0, 1, 1000), simons_F_inv, label="Simons et al. (2022)")
ax.set_xlabel(r"$F(s)$")
ax.set_ylabel(r"$\log_{10}(s)$")
fig.savefig("simons_dfe_F_inv.pdf", bbox_inches="tight")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.hist(s_sample, bins=100, density=True)
ax.plot(s_set, simons_f)
ax.set_xlabel(r"$\log_{10}(s_{ud})$")
ax.set_ylabel(r"$f(\log_{10}(s_{ud}))$")
fig.savefig("simons_dfe_sample.pdf", bbox_inches="tight")
