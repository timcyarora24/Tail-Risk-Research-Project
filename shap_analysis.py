

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables',  exist_ok=True)

TARGET_ASSETS = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']
HORIZONS      = ['1w', '1m', '3m']

# load rolling importances saved by model_training.py 
print("Loading rolling feature importances (in-sample) ...")
importance_dict = {}
missing = []

for asset in TARGET_ASSETS:
    for horizon in HORIZONS:
        task = f'{asset}_q05_{horizon}'
        path = f'results/tables/rolling_importance_{task}.csv'
        try:
            df = pd.read_csv(path).set_index('feature')['importance']
            importance_dict[task] = df
            print(f"  {task:35s}  {len(df)} features")
        except FileNotFoundError:
            missing.append(task)

if missing:
    print(f"\nWARNING: {len(missing)} importance files not found:")
    for m in missing:
        print(f"  {m}")
    print("  Re-run model_training.py to generate them.")

if not importance_dict:
    print("No importance files found. Run model_training.py first.")
    exit(1)

print(f"\nLoaded {len(importance_dict)} importance files.")


# Categorize CASSI features into economic groups
def categorize(feature_name):
    f = feature_name.lower()
    if 'corr' in f:
        return 'Cross-Asset Correlations'
    if 'treasury' in f or 'yield_curve' in f:
        return 'Bond Market'
    if 'credit' in f:
        return 'Credit Market'
    if 'eurusd' in f or 'dollar' in f or 'jpyusd' in f or 'safe_haven' in f:
        return 'Currency / FX'
    if 'gold' in f or 'copper' in f or 'oil' in f:
        return 'Commodities'
    if 'sp500' in f or 'stock_bond' in f:
        return 'Equity'
    return 'Other'

CATEGORY_COLORS = {
    'Cross-Asset Correlations': '#2196F3',
    'Bond Market'             : '#4CAF50',
    'Credit Market'           : '#F44336',
    'Currency / FX'           : '#FF9800',
    'Commodities'             : '#9C27B0',
    'Equity'                  : '#00BCD4',
    'Other'                   : '#9E9E9E',
}

# FIGURE 1: Top 15 features per asset (averaged across horizons)
print("\nGenerating Figure 1: Feature importance by asset ...")

fig, axes = plt.subplots(len(TARGET_ASSETS), 1,
                          figsize=(12, 5 * len(TARGET_ASSETS)))

for idx, asset in enumerate(TARGET_ASSETS):
    tasks = [f'{asset}_q05_{h}' for h in HORIZONS
             if f'{asset}_q05_{h}' in importance_dict]

    if not tasks:
        continue

    avg_imp = pd.concat([importance_dict[t] for t in tasks], axis=1).mean(axis=1)
    avg_imp = avg_imp.sort_values(ascending=True).tail(15)

    ax = axes[idx]
    colors = [CATEGORY_COLORS.get(categorize(f), '#9E9E9E')
              for f in avg_imp.index]

    bars = ax.barh(range(len(avg_imp)), avg_imp.values,
                   color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(avg_imp)))
    ax.set_yticklabels(avg_imp.index, fontsize=9)
    ax.set_title(f'{asset.upper()} — Top 15 Features (avg across 1W/1M/3M)',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Feature Importance (Gain, in-sample avg)', fontsize=9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=cat)
                   for cat, c in CATEGORY_COLORS.items()
                   if cat != 'Other']
fig.legend(handles=legend_elements, loc='lower center',
           ncol=3, fontsize=9, title='Feature Category',
           bbox_to_anchor=(0.5, -0.01))

plt.suptitle('CASSI Feature Importance by Asset\n(Rolling In-Sample, LGB-CASSI)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/figures/feature_importance_by_asset.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/feature_importance_by_asset.png")

# FIGURE 2: Heatmap — top 20 features across all 15 tasks
print("Generating Figure 2: Feature importance heatmap ...")

imp_matrix = pd.DataFrame(importance_dict).fillna(0)


imp_matrix = imp_matrix.div(imp_matrix.sum(axis=0), axis=1) * 100


top_features = imp_matrix.mean(axis=1).sort_values(ascending=False).head(20).index
imp_top = imp_matrix.loc[top_features]

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(imp_top.values, aspect='auto', cmap='YlOrRd')

ax.set_xticks(range(len(imp_top.columns)))
ax.set_xticklabels(imp_top.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(imp_top.index)))
ax.set_yticklabels(imp_top.index, fontsize=9)


for i, feat in enumerate(imp_top.index):
    cat   = categorize(feat)
    color = CATEGORY_COLORS.get(cat, '#9E9E9E')
    ax.get_yticklabels()[i].set_color(color)

plt.colorbar(im, ax=ax, label='% of total importance', shrink=0.8)
ax.set_title('Top 20 Feature Importance Heatmap\n(% of total gain, LGB-CASSI, in-sample)',
             fontsize=12, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('results/figures/feature_importance_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/feature_importance_heatmap.png")

# FIGURE 3: Category breakdown stacked bar 
print("Generating Figure 3: Category breakdown ...")

cat_imp = {}
for task, imp in importance_dict.items():
    total = imp.sum()
    if total == 0:
        continue
    row = {}
    for feat, val in imp.items():
        cat = categorize(feat)
        row[cat] = row.get(cat, 0) + val / total * 100
    cat_imp[task] = row

cat_df = pd.DataFrame(cat_imp).T.fillna(0)
cat_df = cat_df.reindex(columns=[c for c in CATEGORY_COLORS if c in cat_df.columns])

fig, ax = plt.subplots(figsize=(16, 6))
bottom = np.zeros(len(cat_df))
for cat in cat_df.columns:
    vals = cat_df[cat].values
    ax.bar(range(len(cat_df)), vals, bottom=bottom,
           label=cat, color=CATEGORY_COLORS.get(cat, '#9E9E9E'),
           edgecolor='white', linewidth=0.3)
    bottom += vals

ax.set_xticks(range(len(cat_df)))
ax.set_xticklabels(cat_df.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('% of total importance')
ax.set_title('Feature Category Importance by Task\n(LGB-CASSI, rolling in-sample)',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('results/figures/feature_category_breakdown.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/feature_category_breakdown.png")

# Category importance table 
cat_df.to_csv('results/tables/feature_category_importance.csv')
print("  Saved: results/tables/feature_category_importance.csv")

# Print summary 
print("\n" + "="*65)
print("FEATURE CATEGORY SUMMARY (avg % importance across all tasks)")
print("="*65)
avg_cat = cat_df.mean().sort_values(ascending=False)
for cat, val in avg_cat.items():
    bar = '█' * int(val / 2)
    print(f"  {cat:30s}  {val:5.1f}%  {bar}")

print("\n" + "="*65)
print("TOP 5 FEATURES OVERALL (avg across all 15 tasks)")
print("="*65)
overall_imp = pd.DataFrame(importance_dict).mean(axis=1).sort_values(ascending=False)
for feat, val in overall_imp.head(10).items():
    cat = categorize(feat)
    print(f"  {feat:35s}  {val:8.1f}  [{cat}]")

print("\nAll figures saved to results/figures/")