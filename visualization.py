

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/figures', exist_ok=True)

metrics  = pd.read_csv('results/tables/comparison_metrics.csv')
kupiec   = pd.read_csv('results/tables/kupiec_results.csv')
crisis   = pd.read_csv('results/tables/crisis_analysis.csv')


try:
    christ = pd.read_csv('results/tables/christoffersen_results.csv')
    HAS_CC = True
except FileNotFoundError:
    HAS_CC = False
    print("christoffersen_results.csv not found — skipping CC plots")


try:
    mcs_df = pd.read_csv('results/tables/mcs_results.csv')
    HAS_MCS = True
except FileNotFoundError:
    HAS_MCS = False

MODEL_LABELS = {
    'LASSO_AR'   : 'LASSO-AR',
    'LASSO_VIX'  : 'LASSO-VIX',
    'LASSO_CASSI': 'LASSO-CASSI',
    'LGB_VIX'    : 'LGB-VIX',
    'LGB_CASSI'  : 'LGB-CASSI',
}
MODEL_COLORS = {
    'LASSO_AR'   : '#9E9E9E',
    'LASSO_VIX'  : '#2196F3',
    'LASSO_CASSI': '#03A9F4',
    'LGB_VIX'    : '#FF9800',
    'LGB_CASSI'  : '#F44336',
}
ASSETS    = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']
HORIZONS  = ['1w', '1m', '3m']

print("Generating publication figures ...")

#  FIG 1: Quantile loss comparison 
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
ql_cols = {k: f'QL_{k}' for k in MODEL_LABELS}

for hi, horizon in enumerate(HORIZONS):
    ax   = axes[hi]
    sub  = metrics[metrics['Horizon'] == horizon]
    x    = np.arange(len(ASSETS))
    w    = 0.15
    offsets = np.linspace(-2*w, 2*w, 5)

    for mi, (mkey, mlabel) in enumerate(MODEL_LABELS.items()):
        col = ql_cols[mkey]
        if col not in sub.columns:
            continue
        vals = [sub[sub['Asset']==a][col].values[0]
                if len(sub[sub['Asset']==a]) > 0 else 0 for a in ASSETS]
        ax.bar(x + offsets[mi], vals, w*0.9,
               label=mlabel, color=MODEL_COLORS[mkey],
               edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_',' ').title() for a in ASSETS],
                        rotation=30, ha='right', fontsize=9)
    ax.set_title(f'Horizon: {horizon.upper()}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quantile Loss (lower = better)', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [mpatches.Patch(color=MODEL_COLORS[k], label=MODEL_LABELS[k])
           for k in MODEL_LABELS]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))
plt.suptitle('Quantile Loss — All 5 Models × 3 Horizons',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig1_quantile_loss.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 1 saved: fig1_quantile_loss.png")

# FIG 2: Calibration heatmap (Kupiec + CC) 
models_order = list(MODEL_LABELS.keys())
tasks = [f'{a}_{h}' for a in ASSETS for h in HORIZONS]

def make_cal_matrix(df, result_col):
    mat = pd.DataFrame(index=models_order, columns=tasks)
    for _, row in df.iterrows():
        task = f"{row['Asset']}_{row['Horizon']}"
        model = row['Model']
        if model in mat.index and task in mat.columns:
            mat.loc[model, task] = row[result_col]
    return mat

kup_col = 'Kupiec' if 'Kupiec' in kupiec.columns else 'Calibration'
kup_mat = make_cal_matrix(kupiec, kup_col)

def result_to_num(val):
    if val == 'PASS': return 1
    if val == 'FAIL': return -1
    return 0

kup_num = kup_mat.map(result_to_num).astype(float)

n_rows = 2 if HAS_CC else 1
fig, axes = plt.subplots(n_rows, 1,
                          figsize=(18, 4*n_rows + 1))
if n_rows == 1:
    axes = [axes]

for ax, (mat_num, title) in zip(axes, [
    (kup_num, 'Kupiec POF Test — Violation Frequency Calibration'),
]):
    im = ax.imshow(mat_num.values, aspect='auto',
                   cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models_order], fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')

    for i in range(len(models_order)):
        for j in range(len(tasks)):
            val = mat_num.values[i, j]
            text = 'PASS' if val == 1 else ('FAIL' if val == -1 else 'n/a')
            color = 'white' if abs(val) == 1 else 'gray'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')

if HAS_CC:
    cc_mat = make_cal_matrix(christ, 'CC_Test')
    cc_num = cc_mat.map(result_to_num).astype(float)
    ax2 = axes[1]
    ax2.imshow(cc_num.values, aspect='auto',
               cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
    ax2.set_xticks(range(len(tasks)))
    ax2.set_xticklabels(tasks, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(len(models_order)))
    ax2.set_yticklabels([MODEL_LABELS[m] for m in models_order], fontsize=10)
    ax2.set_title('Christoffersen Independence Test — Violation Clustering',
                  fontsize=11, fontweight='bold')
    for i in range(len(models_order)):
        for j in range(len(tasks)):
            val = cc_num.values[i, j]
            text = 'PASS' if val == 1 else ('FAIL' if val == -1 else 'n/a')
            color = 'white' if abs(val) == 1 else 'gray'
            ax2.text(j, i, text, ha='center', va='center',
                     fontsize=6, color=color, fontweight='bold')

plt.suptitle('VaR Backtest Results', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/figures/fig2_calibration_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 2 saved: fig2_calibration_heatmap.png")

#  FIG 3: DM test p-value heatmap 
dm_pairs = {
    'Q1: LGB-VIX\nvs LASSO-AR' : 'DM_VIX_vs_AR_p',
    'Q2: LASSO-CASSI\nvs LASSO-VIX': 'DM_LASSO_CASSI_VIX_p',
    'Q3: LGB-CASSI\nvs LGB-VIX'  : 'DM_CASSI_vs_VIX_p',
    'Q4: LGB-VIX\nvs LASSO-VIX' : 'DM_NLR_p',
}
# fallback for old column names
if 'DM_NLR_p' not in metrics.columns and 'DM_LGB_vs_LASSO_p' in metrics.columns:
    dm_pairs['Q4: LGB-VIX\nvs LASSO-VIX'] = 'DM_LGB_vs_LASSO_p'

task_labels = [f"{r['Asset']}\n{r['Horizon']}" for _, r in metrics.iterrows()]
dm_matrix   = np.full((len(dm_pairs), len(metrics)), np.nan)

for ri, (label, col) in enumerate(dm_pairs.items()):
    if col in metrics.columns:
        dm_matrix[ri] = metrics[col].values

fig, ax = plt.subplots(figsize=(18, 5))
im = ax.imshow(dm_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.2)

ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(task_labels, fontsize=7)
ax.set_yticks(range(len(dm_pairs)))
ax.set_yticklabels(list(dm_pairs.keys()), fontsize=10)

for ri in range(len(dm_pairs)):
    for ci in range(len(metrics)):
        val = dm_matrix[ri, ci]
        if not np.isnan(val):
            sig = '**' if val < 0.01 else ('*' if val < 0.05 else f'{val:.2f}')
            color = 'white' if val < 0.05 else 'black'
            ax.text(ci, ri, sig, ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='p-value', shrink=0.8)
ax.set_title('Diebold-Mariano Test p-Values\n'
             '(** p<0.01, * p<0.05, darker = more significant)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/fig3_dm_pvalues.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 3 saved: fig3_dm_pvalues.png")

# FIG 4: Crisis vs Normal improvement 
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for hi, horizon in enumerate(HORIZONS):
    ax  = axes[hi]
    sub = crisis[crisis['Horizon'] == horizon]

    normal  = sub[sub['Regime'] == 'Normal'].set_index('Asset')['CASSI_Imp_%']
    stress  = sub[sub['Regime'] == 'Stress_VIX20'].set_index('Asset')['CASSI_Imp_%']
    hstress = sub[sub['Regime'] == 'HighStr_VIX30'].set_index('Asset')['CASSI_Imp_%']

    x = np.arange(len(ASSETS))
    w = 0.25

    for vals, offset, color, label in [
        (normal,  -w,   '#2196F3', 'Normal (VIX≤20)'),
        (stress,   0,   '#FF9800', 'Stress (VIX>20)'),
        (hstress,  w,   '#F44336', 'High Stress (VIX>30)'),
    ]:
        v = [vals.get(a, 0) for a in ASSETS]
        bars = ax.bar(x + offset, v, w*0.9,
                      label=label, color=color,
                      edgecolor='white', linewidth=0.3)
        for bar, val in zip(bars, v):
            if abs(val) > 2:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.5 if val > 0 else -1.5),
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=6, color='black')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_',' ').title() for a in ASSETS],
                        rotation=30, ha='right', fontsize=9)
    ax.set_title(f'{horizon.upper()} Horizon', fontsize=11, fontweight='bold')
    ax.set_ylabel('CASSI Improvement over VIX (%)', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))
plt.suptitle('LGB-CASSI vs LGB-VIX: Improvement by Market Regime',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig4_crisis_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 4 saved: fig4_crisis_analysis.png")

# FIG 5: Violation rates — actual vs expected 5% 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for hi, horizon in enumerate(HORIZONS):
    ax  = axes[hi]
    sub = kupiec[kupiec['Horizon'] == horizon]
    x   = np.arange(len(ASSETS))
    w   = 0.15
    offsets = np.linspace(-2*w, 2*w, 5)

    for mi, (mkey, mlabel) in enumerate(MODEL_LABELS.items()):
        msub = sub[sub['Model'] == mkey].set_index('Asset')
        vals = [msub.loc[a, 'Violation_Rate_%']
                if a in msub.index else 5.0 for a in ASSETS]
        ax.bar(x + offsets[mi], vals, w*0.9,
               label=mlabel, color=MODEL_COLORS[mkey],
               edgecolor='white', linewidth=0.3, alpha=0.85)

    ax.axhline(5.0, color='black', linewidth=1.5,
               linestyle='--', label='Target 5%', zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_',' ').title() for a in ASSETS],
                        rotation=30, ha='right', fontsize=9)
    ax.set_title(f'{horizon.upper()} Horizon', fontsize=11, fontweight='bold')
    ax.set_ylabel('Violation Rate (%)', fontsize=9)
    ax.set_ylim(0, max(sub['Violation_Rate_%'].max() * 1.2, 8))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.suptitle('Violation Rates by Model (dashed = target 5%)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig5_violation_rates.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 5 saved: fig5_violation_rates.png")

print("\n" + "="*65)
print("All figures saved to results/figures/")
print("="*65)