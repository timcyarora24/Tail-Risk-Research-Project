
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/tables',  exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


print("Loading data...")

def load_pred(fname):
    df = pd.read_csv(f'results/predictions/{fname}', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df

actual         = load_pred('actual_targets.csv')
pred_lgb_vix   = load_pred('vix_predictions.csv')
pred_lgb_cassi = load_pred('cassi_predictions.csv')


vix_raw  = pd.read_csv('data/raw/vix.csv', header=0)
date_col = vix_raw.columns[0]
junk     = {date_col, 'Ticker', 'Price', 'Date', 'date'}
vix_raw  = vix_raw[~vix_raw[date_col].isin(junk)]
vix_raw[date_col] = pd.to_datetime(vix_raw[date_col], errors='coerce')
vix_raw  = vix_raw.dropna(subset=[date_col]).set_index(date_col)
vix_raw.index = vix_raw.index.normalize()

if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = ['_'.join([str(c) for c in col if c]).strip('_')
                       for col in vix_raw.columns]

close_cands = [c for c in vix_raw.columns if 'close' in c.lower()]
vix_close   = pd.to_numeric(
    vix_raw[close_cands[0]] if close_cands else vix_raw.iloc[:, 0],
    errors='coerce'
).dropna()


CRISIS_PERIODS = {
    'GFC\n2008-09': ('2008-09-01', '2009-06-30'),
    'Euro\nDebt\n2011': ('2011-07-01', '2012-03-31'),
    'COVID\n2020': ('2020-02-01', '2020-06-30'),
    'Rate\nHike\n2022': ('2022-01-01', '2022-12-31'),
}

TARGET_ASSETS = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']
HORIZONS      = ['1w', '1m', '3m']
ALPHA         = 0.05


def get_violations(y_true, y_pred):
    """Returns boolean Series: True where actual < predicted quantile."""
    common = y_true.index.intersection(y_pred.index)
    y = y_true.loc[common]
    q = y_pred.loc[common]
    valid = y.notna() & q.notna()
    viol = pd.Series(False, index=common)
    viol[valid] = y[valid] < q[valid]
    return viol


print("\n" + "="*70)
print("ANALYSIS 1: Violation rate by VIX regime")
print("="*70)

clustering_rows = []

for asset in TARGET_ASSETS:
    for horizon in HORIZONS:
        task = f'{asset}_q05_{horizon}'
        if task not in actual.columns:
            continue

        y = actual[task].dropna()

        for model_name, pred_df in [('LGB_VIX', pred_lgb_vix),
                                     ('LGB_CASSI', pred_lgb_cassi)]:
            if task not in pred_df.columns:
                continue

            viol = get_violations(y, pred_df[task])
            viol_dates = viol.index

            
            vix_aligned = vix_close.reindex(viol_dates, method='ffill')

            regimes = {
                'All':         viol_dates,
                'Normal':      viol_dates[vix_aligned <= 20],
                'Stress':      viol_dates[vix_aligned > 20],
                'HighStress':  viol_dates[vix_aligned > 30],
            }

            row = {'Asset': asset, 'Horizon': horizon, 'Model': model_name}
            for regime_name, idx in regimes.items():
                if len(idx) < 5:
                    row[f'ViolRate_{regime_name}'] = np.nan
                    row[f'N_{regime_name}'] = len(idx)
                    continue
                v = viol.reindex(idx).fillna(False)
                row[f'ViolRate_{regime_name}'] = round(v.mean() * 100, 2)
                row[f'N_{regime_name}'] = len(idx)

            clustering_rows.append(row)

clustering_df = pd.DataFrame(clustering_rows)
clustering_df.to_csv('results/tables/violation_clustering.csv', index=False)
print(clustering_df[['Asset','Horizon','Model',
                      'ViolRate_All','ViolRate_Normal',
                      'ViolRate_Stress','ViolRate_HighStress']].to_string(index=False))


print("\n" + "="*70)
print("ANALYSIS 2: Violation lead time before VIX spikes")
print("  Positive = model flags tail risk BEFORE VIX crosses 20")
print("="*70)


vix_series    = vix_close.sort_index()
vix_above20   = (vix_series > 20).astype(int)
vix_crossings = vix_series[(vix_above20.diff() == 1)]   # first day VIX > 20

lead_rows = []

for asset in TARGET_ASSETS:
    for horizon in ['1w', '1m']:   
        task = f'{asset}_q05_{horizon}'
        if task not in actual.columns:
            continue

        y = actual[task].dropna()

        for model_name, pred_df in [('LGB_VIX',   pred_lgb_vix),
                                     ('LGB_CASSI', pred_lgb_cassi)]:
            if task not in pred_df.columns:
                continue

            viol = get_violations(y, pred_df[task])

            for cross_date in vix_crossings.index:
                
                window_start = cross_date - pd.Timedelta(days=30)
                window_end   = cross_date + pd.Timedelta(days=10)
                window_viol  = viol[(viol.index >= window_start) &
                                    (viol.index <= window_end)]

                if window_viol.sum() == 0:
                    continue

                
                first_viol = window_viol[window_viol].index[0]
                lead_days  = (cross_date - first_viol).days  

                lead_rows.append({
                    'Asset'      : asset,
                    'Horizon'    : horizon,
                    'Model'      : model_name,
                    'CrossDate'  : cross_date.date(),
                    'FirstViol'  : first_viol.date(),
                    'LeadDays'   : lead_days,
                    'VIX_at_cross': round(vix_series.get(cross_date, np.nan), 1),
                })

if lead_rows:
    lead_df = pd.DataFrame(lead_rows)
    lead_df.to_csv('results/tables/violation_lead_time.csv', index=False)

    summary = (lead_df.groupby(['Model', 'Horizon'])['LeadDays']
               .agg(['mean', 'median', 'std', 'count'])
               .round(1))
    print(summary.to_string())

   
    for horizon in ['1w', '1m']:
        sub = lead_df[lead_df['Horizon'] == horizon]
        vix_lead   = sub[sub['Model'] == 'LGB_VIX']['LeadDays'].mean()
        cassi_lead = sub[sub['Model'] == 'LGB_CASSI']['LeadDays'].mean()
        print(f"\n  {horizon}: LGB-VIX avg lead = {vix_lead:.1f} days | "
              f"LGB-CASSI avg lead = {cassi_lead:.1f} days | "
              f"CASSI advantage = {cassi_lead - vix_lead:+.1f} days")
else:
    print("  No lead-time data computed (check prediction/actual alignment)")
    lead_df = pd.DataFrame()


print("\n" + "="*70)
print("ANALYSIS 3: Generating violation clustering figure")
print("="*70)


task   = 'sp500_q05_1w'
y      = actual[task].dropna()

viol_vix   = get_violations(y, pred_lgb_vix[task])
viol_cassi = get_violations(y, pred_lgb_cassi[task])
vix_plot   = vix_close.reindex(y.index, method='ffill')

fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                         gridspec_kw={'height_ratios': [2, 1, 1, 1]})
fig.suptitle('Violation Clustering Analysis: S&P 500 1-Week Tail Risk\n'
             'Kupiec failures concentrated in crisis regimes — not random miscalibration',
             fontsize=13, fontweight='bold', y=0.98)


ax = axes[0]
ax.plot(vix_plot.index, vix_plot.values, color='#333333', lw=0.8, label='VIX')
ax.axhline(20, color='orange', ls='--', lw=1.2, label='VIX=20 (stress)')
ax.axhline(30, color='red',    ls='--', lw=1.2, label='VIX=30 (high stress)')
ax.fill_between(vix_plot.index, 20, vix_plot.values,
                where=vix_plot.values > 20, alpha=0.15, color='orange')
ax.fill_between(vix_plot.index, 30, vix_plot.values,
                where=vix_plot.values > 30, alpha=0.25, color='red')
for label, (s, e) in CRISIS_PERIODS.items():
    try:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.08, color='navy', zorder=0)
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 60,
                label, ha='center', fontsize=7, color='navy', va='top')
    except Exception:
        pass
ax.set_ylabel('VIX Level')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(y.index[0], y.index[-1])


ax = axes[1]
viol_v_dates = viol_vix[viol_vix].index
ax.scatter(viol_v_dates, np.ones(len(viol_v_dates)),
           s=4, color='#FF9800', alpha=0.6, label='Violation')
# Rolling violation rate (63-day)
roll_viol_vix = viol_vix.rolling(63, min_periods=20).mean() * 100
ax.plot(roll_vix := roll_viol_vix.index, roll_viol_vix.values,
        color='#FF9800', lw=1.5, label='63d violation rate %')
ax.axhline(5, color='black', ls=':', lw=1, label='Target 5%')
ax.fill_between(vix_plot.index, 0, 1,
                where=vix_plot.values > 20, alpha=0.1, color='orange')
ax.set_ylabel('LGB-VIX\nViolation Rate %')
ax.set_ylim(0, None)
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(y.index[0], y.index[-1])


viol_c_dates = viol_cassi[viol_cassi].index
ax.scatter(viol_c_dates, np.ones(len(viol_c_dates)),
           s=4, color='#F44336', alpha=0.6, label='Violation')
roll_viol_cassi = viol_cassi.rolling(63, min_periods=20).mean() * 100
ax.plot(roll_viol_cassi.index, roll_viol_cassi.values,
        color='#F44336', lw=1.5, label='63d violation rate %')
ax.axhline(5, color='black', ls=':', lw=1, label='Target 5%')
ax.fill_between(vix_plot.index, 0, 1,
                where=vix_plot.values > 20, alpha=0.1, color='orange')
ax.set_ylabel('LGB-CASSI\nViolation Rate %')
ax.set_ylim(0, None)
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(y.index[0], y.index[-1])

ax = axes[3]
diff = roll_viol_cassi - roll_viol_vix.reindex(roll_viol_cassi.index)
ax.bar(diff.index, diff.values, width=1,
       color=np.where(diff.values > 0, '#F44336', '#4CAF50'), alpha=0.6)
ax.axhline(0, color='black', lw=1)
ax.set_ylabel('CASSI − VIX\nViol. Rate (pp)')
ax.set_xlabel('Date')
ax.fill_between(vix_plot.index, diff.min() if len(diff.dropna()) > 0 else -5,
                diff.max() if len(diff.dropna()) > 0 else 5,
                where=vix_plot.values > 20, alpha=0.1, color='orange')
ax.set_xlim(y.index[0], y.index[-1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/figures/fig_violation_clustering.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: results/figures/fig_violation_clustering.png")

print("\n" + "="*70)
print("KEY STATS FOR PAPER (copy these into your results section)")
print("="*70)

for asset in TARGET_ASSETS:
    for horizon in HORIZONS:
        task = f'{asset}_q05_{horizon}'
        if task not in actual.columns:
            continue
        y = actual[task].dropna()

        for model_name, pred_df in [('LGB_VIX', pred_lgb_vix),
                                     ('LGB_CASSI', pred_lgb_cassi)]:
            if task not in pred_df.columns:
                continue
            viol       = get_violations(y, pred_df[task])
            vix_aln    = vix_close.reindex(viol.index, method='ffill')
            normal_vr  = viol[vix_aln <= 20].mean() * 100
            stress_vr  = viol[vix_aln > 20].mean()  * 100
            ratio      = stress_vr / normal_vr if normal_vr > 0 else np.nan
            print(f"  {asset:15s} {horizon} {model_name:12s} | "
                  f"Normal={normal_vr:.1f}%  Stress={stress_vr:.1f}%  "
                  f"Ratio={ratio:.1f}x")

print("\nDone. Files saved:")
print("  results/tables/violation_clustering.csv")
print("  results/tables/violation_lead_time.csv")
print("  results/figures/fig_violation_clustering.png")
