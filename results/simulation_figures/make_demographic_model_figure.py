import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import smilenfer.plotting as splot
import smilenfer.simulation as sim


splot._plot_params()
matplotlib.rcParams.update({"font.size": 18})

tenn_n = sim.tennessen_model()
joug_n = sim.jouganous_model_jpt()

tenn_x = np.flip(np.arange(len(tenn_n)))
joug_x = np.flip(np.arange(len(joug_n)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(tenn_x, tenn_n, lw=2.5, color="#0072B2")
axes[0].set_xlim((len(tenn_n) + 100, -100))
axes[0].set_yscale("log")
axes[0].set_xlabel("Generations ago")
axes[0].set_ylabel("Population size")
axes[0].set_title("Tennessen EUR")

axes[1].plot(joug_x, joug_n, lw=2.5, color="#D55E00")
axes[1].set_xlim((len(joug_n) + 100, -100))
axes[1].set_yscale("log")
axes[1].set_xlabel("Generations ago")
axes[1].set_ylabel("Population size")
axes[1].set_title("Jouganous JPT")

fig.tight_layout()
fig.savefig("supp_demographic_models.pdf", bbox_inches="tight")
