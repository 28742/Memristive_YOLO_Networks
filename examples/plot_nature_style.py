import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall style for Nature/Science look
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 12,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def plot_publication_quality():
    # Read the data
    df = pd.read_csv("diff_pair_vs_default_results.csv")
    
    # Strip whitespace from columns just in case
    df.columns = df.columns.str.strip()
    df["Architecture"] = df["Architecture"].str.strip()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Define colors (Colorblind friendly & professional)
    color_default = "#E64B35"  # Red-ish
    color_diff = "#4DBBD5"     # Blue-ish
    
    # Plot mAP50-95
    for arch, color, marker in zip(["Default", "Differential Pair"], [color_default, color_diff], ['o', 's']):
        subset = df[df["Architecture"] == arch]
        ax1.plot(subset["Write Variation"], subset["mAP50-95"], 
                 marker=marker, color=color, label=arch, 
                 markeredgecolor='black', markeredgewidth=0.8, alpha=0.9)
    
    ax1.set_xlabel("Write Variation ($\sigma$)")
    ax1.set_ylabel("mAP 50-95")
    ax1.set_ylim(0, 0.35)
    ax1.set_xticks([0, 0.02, 0.05, 0.07, 0.1])
    # ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    
    # Plot mAP50
    for arch, color, marker in zip(["Default", "Differential Pair"], [color_default, color_diff], ['o', 's']):
        subset = df[df["Architecture"] == arch]
        ax2.plot(subset["Write Variation"], subset["mAP50"], 
                 marker=marker, color=color, label=arch, 
                 markeredgecolor='black', markeredgewidth=0.8, alpha=0.9)
    
    ax2.set_xlabel("Write Variation ($\sigma$)")
    ax2.set_ylabel("mAP 50")
    ax2.set_ylim(0, 0.55)
    ax2.set_xticks([0, 0.02, 0.05, 0.07, 0.1])
    # ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Despine top and right axes for that clean Nature look
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Add legend to the first plot (or a shared one)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
               ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.savefig("nature_style_plot.pdf")
    plt.savefig("nature_style_plot.png")
    print("Saved publication quality plots: nature_style_plot.pdf and nature_style_plot.png")

if __name__ == "__main__":
    plot_publication_quality()
