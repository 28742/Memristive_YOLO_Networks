import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create directory to save figures
save_dir = "paper_figures"
os.makedirs(save_dir, exist_ok=True)

# Define color palette academic style
COLOR_MAIN = "#2b5797"
COLOR_ACCENT = "#e04006"
COLOR_SSOR = "#1d8038" # Green
COLOR_DIFF = "#c0392b" # Red
COLOR_BG = "#ecf0f1"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ==============================================================================
# Figure 1: Motivation (Area vs Conductance Collapse)
# ==============================================================================
def plot_fig1_motivation():
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    methods = ['Two\'s Complement', 'Differential Pair', 'Ours: SSOR']
    # Simulated normalized values
    energy = [10.0, 0.5, 0.3]   # Conductance (Leakage) when mapping weight = -1
    area = [1.0, 2.0, 1.05]     # Physical crossbar overhead 
    
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, area, width, label='Hardware Area Overhead (Lower is better)', color=COLOR_MAIN, alpha=0.8)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, energy, width, label='Static Conductance Energy (Weight=-1)', color=COLOR_ACCENT, alpha=0.8)
    
    ax1.set_ylabel('Normalized Physical Area', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Standby Conductance / Power Leakage', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 1: Comparison of Mapping Extremely Small Negative Weights (e.g. -1)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    
    # Grid and styling
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Figure1_Motivation.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "Figure1_Motivation.pdf"), bbox_inches='tight')
    plt.close()

# ==============================================================================
# Figure 2: SSOR Micro-Architecture
# ==============================================================================
def plot_fig2_ssor_arch():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Draw Main Array (W_mapped)
    ax.add_patch(patches.Rectangle((0.1, 0.4), 0.4, 0.5, edgecolor=COLOR_MAIN, facecolor='#d6eaf8', lw=2, zorder=2))
    ax.text(0.3, 0.65, "Main ReRAM Array\n(Positive Weights: W+Z)", ha='center', va='center', fontsize=12, fontweight='bold', color='#154360', zorder=3)
    
    # Draw Dummy Column (Z)
    ax.add_patch(patches.Rectangle((0.55, 0.4), 0.1, 0.5, edgecolor=COLOR_ACCENT, facecolor='#fadbd8', lw=2, zorder=2))
    ax.text(0.6, 0.65, "Dummy\n(Z)\n\nZ*sum(X)", ha='center', va='center', fontsize=11, fontweight='bold', color='#641e16', zorder=3)
    
    # Inputs
    ax.annotate('', xy=(0.1, 0.8), xytext=(-0.05, 0.8), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.annotate('', xy=(0.1, 0.5), xytext=(-0.05, 0.5), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(-0.02, 0.65, "Wordlines\n(Inputs X)", ha='center', va='center', fontsize=11)
    
    # Output to ADCs
    ax.add_patch(patches.Rectangle((0.1, 0.2), 0.4, 0.1, edgecolor='gray', facecolor='lightgray', lw=2))
    ax.text(0.3, 0.25, "ADC Array", ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.add_patch(patches.Rectangle((0.55, 0.2), 0.1, 0.1, edgecolor='gray', facecolor='lightgray', lw=2))
    ax.text(0.6, 0.25, "ADC", ha='center', va='center', fontsize=11, fontweight='bold')
    
    for i in np.arange(0.15, 0.5, 0.1):
        ax.annotate('', xy=(i, 0.3), xytext=(i, 0.4), arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))
    ax.annotate('', xy=(0.6, 0.3), xytext=(0.6, 0.4), arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))
    
    # Digital Subtraction
    ax.add_patch(patches.Rectangle((0.1, 0.05), 0.55, 0.1, edgecolor=COLOR_SSOR, facecolor='#d5f5e3', lw=2))
    ax.text(0.375, 0.1, "Digital Subtraction (Out = Main - Dummy)", ha='center', va='center', fontsize=12, fontweight='bold', color='#145a32')
    
    ax.annotate('', xy=(0.3, 0.15), xytext=(0.3, 0.2), arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))
    ax.annotate('', xy=(0.6, 0.15), xytext=(0.6, 0.2), arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))

    # Polarity Inverter
    ax.add_patch(patches.Rectangle((0.7, 0.05), 0.25, 0.1, edgecolor='purple', facecolor='#e8daef', lw=2))
    ax.text(0.825, 0.1, "Polarity Decoder\n(* Sign_Bit)", ha='center', va='center', fontsize=11, fontweight='bold', color='purple')
    
    ax.annotate('', xy=(0.7, 0.1), xytext=(0.65, 0.1), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.annotate('', xy=(1.0, 0.1), xytext=(0.95, 0.1), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(0.98, 0.15, "Final Output Y", ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.set_title("Figure 2: Micro-Architecture of Sign-Sparse Offset Representation (SSOR)", fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Figure2_SSOR_Architecture.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "Figure2_SSOR_Architecture.pdf"), bbox_inches='tight')
    plt.close()

# ==============================================================================
# Figure 3: MixMap Architecture (YOLOv5 Macro)
# ==============================================================================
def plot_fig3_mixmap_arch():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # MixMap Routing Engine
    ax.add_patch(patches.FancyBboxPatch((0.2, 0.7), 0.6, 0.2, boxstyle="round,pad=0.05", edgecolor='#f39c12', facecolor='#fdf2e9', lw=2))
    ax.text(0.5, 0.8, "MixMap Dynamic Allocator\nDecision: f(Dense[-τ, 0), Sensitivity)", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Mode B Path (SSOR / Single Array)
    ax.add_patch(patches.FancyBboxPatch((0.15, 0.3), 0.25, 0.25, boxstyle="round,pad=0.05", edgecolor=COLOR_SSOR, facecolor='#d5f5e3', lw=2))
    ax.text(0.275, 0.425, "Mode B: SSOR Array\n(Area=1x, Low Density)\n\nApplied to: Backbone (CSP)", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(0.275, 0.55), xytext=(0.4, 0.7), arrowprops=dict(arrowstyle="<-", color=COLOR_SSOR, lw=3, connectionstyle="angle3,angleA=90,angleB=0"))
    
    # Mode A Path (Diff-Pair)
    ax.add_patch(patches.FancyBboxPatch((0.6, 0.3), 0.25, 0.25, boxstyle="round,pad=0.05", edgecolor=COLOR_DIFF, facecolor='#fadbd8', lw=2))
    ax.text(0.725, 0.425, "Mode A: Differential Pair\n(Area=2x, High Noise Risk)\n\nApplied to: Detection Heads", ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(0.725, 0.55), xytext=(0.6, 0.7), arrowprops=dict(arrowstyle="<-", color=COLOR_DIFF, lw=3, connectionstyle="angle3,angleA=90,angleB=180"))
    
    # YOLO blocks mock
    ax.text(0.1, 0.15, "Raw Image", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(0.2, 0.15), xytext=(0.15, 0.15), arrowprops=dict(arrowstyle="->", lw=2))
    
    ax.add_patch(patches.Rectangle((0.2, 0.1), 0.4, 0.1, facecolor='#abebc6', edgecolor='black', lw=1))
    ax.text(0.4, 0.15, "YOLOv5 Backbone & Early Neck", ha='center', va='center')
    
    ax.annotate('', xy=(0.65, 0.15), xytext=(0.6, 0.15), arrowprops=dict(arrowstyle="->", lw=2))
    
    ax.add_patch(patches.Rectangle((0.65, 0.1), 0.2, 0.1, facecolor='#f5b7b1', edgecolor='black', lw=1))
    ax.text(0.75, 0.15, "Prediction\nHeads", ha='center', va='center')
    
    # Connections to Modes
    ax.annotate('', xy=(0.4, 0.2), xytext=(0.3, 0.3), arrowprops=dict(arrowstyle="->", color="black", lw=1.5, linestyle="dashed"))
    ax.annotate('', xy=(0.75, 0.2), xytext=(0.75, 0.3), arrowprops=dict(arrowstyle="->", color="black", lw=1.5, linestyle="dashed"))
    
    ax.set_title("Figure 3: MixMap YOLO Heterogeneous Assignment Subsystem", fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Figure3_MixMap_Architecture.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "Figure3_MixMap_Architecture.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_fig1_motivation()
    plot_fig2_ssor_arch()
    plot_fig3_mixmap_arch()
    print("All figures successfully generated in 'paper_figures' directory.")
