import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
methods = ['SciGraph', 'SciKG', 'SciKGFormer',
           'SciKE', 'SciKGNet', 'Proposed']

precision = [0.891, 0.912, 0.928, 0.941, 0.956, 0.985]
recall    = [0.876, 0.897, 0.916, 0.931, 0.947, 0.972]

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
# -----------------------------
bars1 = ax.bar(x - width/2, precision, width,
               label='Precision',
               color='#4C72B0',
               edgecolor='black',
               linewidth=1)

bars2 = ax.bar(x + width/2, recall, width,
               label='Recall',
               color='#DD8452',
               edgecolor='black',
               linewidth=1)

# -----------------------------
# Labels
# -----------------------------

ax.set_ylabel('Performance', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)

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
plt.savefig('./graph/comp_1.png',
            dpi=600,
            bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
methods = ['SciGraph', 'SciKG', 'SciKGFormer',
           'SciKE', 'SciKGNet', 'Proposed']

precision = [0.883,0.904,0.922,0.936,0.951,0.978]
recall    = [0.872,0.893,0.914,0.928,0.944,0.969]

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
# -----------------------------
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
ax.set_xticklabels(methods, rotation=15)

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
plt.savefig('./graph/comp_2.png',
            dpi=600,
            bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
methods = ['With Preprocess','Without Preprocess','With knowledge extraction','Without knowledge extraction','With Entity Alignment','Without Entity Alignment', 'Proposed']

precision = [0.972,0.949,0.978,0.956,0.981,0.962,0.985]
recall    = [0.958,0.936,0.964,0.943,0.968,0.95,0.972]

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
# -----------------------------
bars1 = ax.bar(x - width/2, precision, width,
               label='Precision',
               color='#4C72B0',
               edgecolor='black',
               linewidth=1)

bars2 = ax.bar(x + width/2, recall, width,
               label='Recall',
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
plt.savefig('./graph/comp_3_abl.png',
            dpi=600,
            bbox_inches='tight')

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
methods = ['With Preprocess','Without Preprocess','With knowledge extraction','Without knowledge extraction','With Entity Alignment','Without Entity Alignment', 'Proposed']

precision = [0.965,0.942,0.971,0.949,0.974,0.956,0.978]
recall    = [0.955,0.932,0.961,0.939,0.965,0.946,0.969]

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
               label='F1-score',
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
plt.savefig('./graph/comp_4_abl.png',
            dpi=600,
            bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
methods = ['5-Fold','10-Fold','15-Fold','20-Fold']

precision = [0.976,0.98,0.983,0.985]
recall    = [0.963,0.967,0.97,0.972]

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
               label='Precision',
               color='#4C72B0',
               edgecolor='black',
               linewidth=1)

bars2 = ax.bar(x + width/2, recall, width,
               label='Recall',
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
plt.savefig('./graph/comp_5_fold.png',
            dpi=600,
            bbox_inches='tight')

plt.show()
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

#  style border
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

# Save figure
plt.savefig('./graph/comp_6_fold.png',
            dpi=600,
            bbox_inches='tight')

plt.show()