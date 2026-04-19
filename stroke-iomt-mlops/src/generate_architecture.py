"""
Generates a professional system architecture diagram for the 
Privacy-Preserving Seizure Prediction System using matplotlib.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, x, y, w, h, text, color, fontsize=8, text_color='white', subtext=None):
    """Draw a rounded rectangle box with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.92
    )
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.03, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, family='sans-serif')
        ax.text(x, y - 0.06, subtext, ha='center', va='center',
                fontsize=fontsize - 2, color=text_color, alpha=0.85, family='sans-serif', style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, family='sans-serif')
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='white'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8, 
                               connectionstyle='arc3,rad=0'))

def generate_architecture_diagram():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.0)
    ax.set_facecolor('#1a1f36')
    fig.patch.set_facecolor('#1a1f36')
    ax.axis('off')

    # Title
    ax.text(0.8, 0.95, 'System Architecture: FedStrokeSeizure-IoMT Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#00d4ff', family='sans-serif')
    ax.text(0.8, 0.91, 'Privacy-Preserving Seizure Prediction using Edge-Fog-Cloud IoMT',
            ha='center', va='center', fontsize=9, color='#8899bb', family='sans-serif')

    # ── LAYER 1: Data Source ──────────────────────
    draw_box(ax, 0.12, 0.60, 0.18, 0.18, 'CHB-MIT EEG\nDataset', '#2563eb',
             fontsize=9, subtext='23 Ch × 256 Hz')
    
    # ── LAYER 2: Preprocessing ────────────────────
    draw_arrow(ax, 0.21, 0.60, 0.30, 0.60, '#4488ff')
    draw_box(ax, 0.38, 0.60, 0.18, 0.18, 'Z-Score\nNormalization', '#059669',
             fontsize=9, subtext='Per-Channel')

    # ── Split into two branches ───────────────────
    # Top branch: Random Forest
    draw_arrow(ax, 0.47, 0.65, 0.56, 0.78, '#ff9933')
    draw_box(ax, 0.66, 0.78, 0.22, 0.12, 'Statistical Feature Extraction', '#d97706',
             fontsize=8, subtext='Mean, Var, Min, Max')
    
    draw_arrow(ax, 0.77, 0.78, 0.87, 0.78, '#ff9933')
    draw_box(ax, 0.98, 0.78, 0.22, 0.12, 'Random Forest Classifier', '#b45309',
             fontsize=8, subtext='100 Trees, Depth=10')

    # Bottom branch: CNN
    draw_arrow(ax, 0.47, 0.55, 0.56, 0.42, '#a855f7')
    draw_box(ax, 0.66, 0.42, 0.22, 0.12, 'Tensor Reshaping', '#7c3aed',
             fontsize=8, subtext='(Batch, TimeSteps, Channels)')
    
    draw_arrow(ax, 0.77, 0.42, 0.87, 0.42, '#a855f7')
    draw_box(ax, 0.98, 0.42, 0.22, 0.14, '1D-CNN Model', '#6d28d9',
             fontsize=8, subtext='Conv1D→ReLU→Pool→Dense→σ')

    # ── LAYER 4: Evaluation ───────────────────────
    draw_arrow(ax, 1.09, 0.78, 1.18, 0.60, '#ef4444')
    draw_arrow(ax, 1.09, 0.42, 1.18, 0.60, '#ef4444')
    draw_box(ax, 1.26, 0.60, 0.18, 0.16, 'Comparative\nEvaluation', '#dc2626',
             fontsize=8, subtext='Acc, P, R, F1, AUC')

    # ── LAYER 5: Deployment (IoMT Stack) ──────────
    draw_arrow(ax, 1.35, 0.60, 1.42, 0.60, '#14b8a6')
    
    # Edge
    draw_box(ax, 1.50, 0.75, 0.14, 0.09, 'Edge Layer', '#0d9488',
             fontsize=7, subtext='TFLite')
    # Fog
    draw_box(ax, 1.50, 0.60, 0.14, 0.09, 'Fog Layer', '#0f766e',
             fontsize=7, subtext='FastAPI')
    # Cloud
    draw_box(ax, 1.50, 0.45, 0.14, 0.09, 'Cloud Layer', '#115e59',
             fontsize=7, subtext='Federated Server')
    
    # Vertical arrows in deployment stack
    draw_arrow(ax, 1.50, 0.705, 1.50, 0.65, '#14b8a6')
    draw_arrow(ax, 1.50, 0.555, 1.50, 0.50, '#14b8a6')

    # ── Legend ────────────────────────────────────
    legend_items = [
        mpatches.Patch(color='#2563eb', label='Data Source'),
        mpatches.Patch(color='#059669', label='Preprocessing'),
        mpatches.Patch(color='#d97706', label='Random Forest Path'),
        mpatches.Patch(color='#7c3aed', label='CNN Path'),
        mpatches.Patch(color='#dc2626', label='Evaluation'),
        mpatches.Patch(color='#0d9488', label='IoMT Deployment'),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=6,
              fontsize=7, facecolor='#1a1f36', edgecolor='#334155',
              labelcolor='white', framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    # ── Layer Labels ──────────────────────────────
    labels = [
        (0.12, 0.85, 'DATA'),
        (0.38, 0.85, 'PREPROCESS'),
        (0.66, 0.88, 'FEATURE ENG.'),
        (0.98, 0.88, 'MODEL'),
        (1.26, 0.85, 'EVALUATE'),
        (1.50, 0.88, 'DEPLOY'),
    ]
    for lx, ly, lt in labels:
        ax.text(lx, ly, lt, ha='center', va='center', fontsize=7,
                color='#64748b', fontweight='bold', family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', 
                         edgecolor='#334155', alpha=0.7))

    plots_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, 'system_architecture.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#1a1f36')
    plt.close()
    print(f"Architecture diagram saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    generate_architecture_diagram()
