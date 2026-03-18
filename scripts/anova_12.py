import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
# Load the statistical features dataset
df = pd.read_csv('features/features_statistical.csv')

# Exclude 'Clara Schumann' due to insufficient sample size; retain only the three core composers
df = df[df['composer'] != 'Clara Schumann'].copy()

# Extract all feature column names (excluding 'filename' and 'composer')
feature_cols = [col for col in df.columns if col not in ['filename', 'composer']]

# Set a unified plot style suitable for academic publications
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14})

# ==========================================
# 2. Core Step: One-Way ANOVA Analysis
# ==========================================
print("=====================================================")
print("  Feature Discriminability Analysis (One-Way ANOVA)  ")
print("=====================================================\n")

anova_results = []
composers = df['composer'].unique()

for feat in feature_cols:
    # Extract data distributions for each composer regarding the current feature
    groups = [df[df['composer'] == comp][feat].values for comp in composers]

    # Perform One-Way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)
    anova_results.append({'Feature': feat, 'F-statistic': f_stat, 'p-value': p_val})

    # Format and display significance levels based on p-value
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"Feature: {feat:<12} | F-stat: {f_stat:>7.3f} | p-val: {p_val:>8.2e} [{significance}]")

print("\n(Note: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant)")
print(
    "Interpretation: A smaller p-value indicates more significant differences between composers, implying higher discriminative power.\n")

# ==========================================
# 3. Visualization: Academic-Style Boxplots
# ==========================================
# Calculate layout dimensions (3 plots per row)
n_features = len(feature_cols)
cols_per_row = 3
rows = int(np.ceil(n_features / cols_per_row))

fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
axes = axes.flatten()  # Flatten the array of axes for easier iteration

for i, feat in enumerate(feature_cols):
    ax = axes[i]

    # Create boxplot using Seaborn with a fixed composer order
    sns.boxplot(x='composer', y=feat, data=df, ax=ax,
                order=['Franz Schubert', 'Robert Schumann', 'Johannes Brahms'])

    # Overlay swarmplot to show individual data point distribution
    # (Adjust alpha/size or comment out if data density causes clutter)
    sns.stripplot(x='composer', y=feat, data=df, ax=ax,
                  order=['Franz Schubert', 'Robert Schumann', 'Johannes Brahms'],
                  color=".25", alpha=0.5, size=3)

    ax.set_title(f"Distribution of {feat}", fontweight='bold')
    ax.set_xlabel("")  # Remove x-label for cleanliness as categories are obvious
    ax.set_ylabel("Value")

    # Slightly rotate x-axis ticks if labels are long
    ax.tick_params(axis='x', rotation=15)

# Remove unused subplots if the total number of features isn't a multiple of 3
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
# Save high-resolution image suitable for publication
plt.savefig("feature_distribution_anova.png", dpi=300, bbox_inches='tight')
print("[+] High-resolution distribution plot saved as 'feature_distribution_anova.png'")
plt.show()