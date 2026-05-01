import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_publication_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 10.8,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
