import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
methods = ['5-Fold','10-Fold','15-Fold','20-Fold']

precision = [0.969,0.973,0.976,0.978]
recall    = [0.96,0.964,0.967,0.969]

# -----------------------------
# IEEE Figure Settings
# -----------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

fig, ax = plt.subplots(figsize=(8,6), dpi=600)

x = np.arange(len(methods))
width = 0.35

# -----------------------------
# Bars
# -----------------------------, 	
	

bars1 = ax.bar(x - width/2, precision, width,
               label='F1-Score',
               color='#4C72B0',
               edgecolor='black',
               linewidth=1)

bars2 = ax.bar(x + width/2, recall, width,
               label='Entity Alignment Accuracy',
               color='#DD8452',
               edgecolor='black',
               linewidth=1)

# -----------------------------
# Labels
# -----------------------------

ax.set_ylabel('Performance', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=25)

ax.set_ylim(0.84, 1.00)

ax.legend(loc='upper left', frameon=True)

# Grid
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Display values
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,
                h+0.002,
                f'{h:.3f}',
                ha='center',
                va='bottom',
                fontsize=12)

# IEEE style border
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

# Save figure
plt.savefig('./graph/comp_6_fold.png',
            dpi=600,
            bbox_inches='tight')

plt.show()