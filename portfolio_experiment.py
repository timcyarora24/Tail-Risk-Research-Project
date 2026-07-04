
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/tables',  exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

#  load data
print("Loading data...")

def load_pred(fname):
    df = pd.read_csv(f'results/predictions/{fname}', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df

pred_lgb_vix   = load_pred('vix_predictions.csv')
pred_lgb_cassi = load_pred('cassi_predictions.csv')
returns_df     = pd.read_csv('data/processed/returns.csv',
                              index_col=0, parse_dates=True)
returns_df.index = pd.to_datetime(returns_df.index).normalize()


TRIGGER_PCT        = 0.20   
REDUCED_EXPOSURE   = 0.30   
FULL_EXPOSURE      = 1.00   
RISK_FREE_DAILY    = 0.04 / 252  

CRISIS_PERIODS = {
    'GFC 2008-09'   : ('2008-09-01', '2009-06-30'),
    'COVID 2020'    : ('2020-02-01', '2020-06-30'),
    'Rate Hike 2022': ('2022-01-01', '2022-12-31'),
}


def compute_metrics(daily_returns, rf_daily=RISK_FREE_DAILY):
    """Compute standard performance metrics from daily return series."""
    r = daily_returns.dropna()
    if len(r) < 30:
        return {}

    ann_ret  = r.mean() * 252
    ann_vol  = r.std()  * np.sqrt(252)
    sharpe   = (r.mean() - rf_daily) / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    # Max drawdown
    cum  = (1 + r).cumprod()
    roll = cum.cummax()
    dd   = (cum - roll) / roll
    mdd  = dd.min()

    # CVaR (Expected Shortfall at 5%)
    cvar = r[r <= r.quantile(0.05)].mean()

    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan

    return {
        'Ann_Return_%'  : round(ann_ret  * 100, 2),
        'Ann_Vol_%'     : round(ann_vol  * 100, 2),
        'Sharpe'        : round(sharpe,          3),
        'Max_Drawdown_%': round(mdd      * 100,  2),
        'CVaR_5pct_%'   : round(cvar     * 100,  2),
        'Calmar'        : round(calmar,           3),
        'N_Days'        : len(r),
    }

def build_managed_portfolio(sp500_returns, signal_series, trigger_pct,
                             full_exp=FULL_EXPOSURE, reduced_exp=REDUCED_EXPOSURE):
    
    common = sp500_returns.index.intersection(signal_series.index)
    r      = sp500_returns.loc[common]
    sig    = signal_series.loc[common]

    
    threshold = sig.rolling(252, min_periods=63).quantile(trigger_pct)

    
    danger = sig < threshold

    # Exposure is set based on PREVIOUS day's signal (no look-ahead)
    exposure = pd.Series(full_exp, index=common)
    exposure[danger.shift(1).fillna(False)] = reduced_exp

    # Portfolio return = exposure * asset_return + (1-exposure) * rf
    port_ret = exposure * r + (1 - exposure) * RISK_FREE_DAILY

    return port_ret, exposure, threshold


print("\n" + "="*70)
print("PORTFOLIO EXPERIMENT: S&P 500")
print("="*70)

sp500_ret = returns_df['sp500'].dropna()

# Use 1-week horizon signal (most actionable)
task = 'sp500_q05_1w'

vix_signal   = pred_lgb_vix[task].dropna()   if task in pred_lgb_vix.columns   else None
cassi_signal = pred_lgb_cassi[task].dropna() if task in pred_lgb_cassi.columns else None

if vix_signal is None or cassi_signal is None:
    print("ERROR: predictions not found. Run model_training.py first.")
    exit(1)


common = sp500_ret.index.intersection(vix_signal.index).intersection(cassi_signal.index)
sp500_common = sp500_ret.loc[common]


ret_bnh = sp500_common  # buy and hold

ret_vix, exp_vix, thr_vix = build_managed_portfolio(
    sp500_common, vix_signal.reindex(common), TRIGGER_PCT)

ret_cassi, exp_cassi, thr_cassi = build_managed_portfolio(
    sp500_common, cassi_signal.reindex(common), TRIGGER_PCT)

print(f"Period: {common[0].date()} to {common[-1].date()}")
print(f"Total days: {len(common)}")
print(f"VIX signal fires:   {(exp_vix < FULL_EXPOSURE).sum()} days "
      f"({(exp_vix < FULL_EXPOSURE).mean()*100:.1f}%)")
print(f"CASSI signal fires: {(exp_cassi < FULL_EXPOSURE).sum()} days "
      f"({(exp_cassi < FULL_EXPOSURE).mean()*100:.1f}%)")


metrics_rows = []
for name, r in [('Buy-and-Hold', ret_bnh),
                ('VIX-Managed',  ret_vix),
                ('CASSI-Managed', ret_cassi)]:
    m = compute_metrics(r)
    m['Strategy'] = name
    metrics_rows.append(m)
    print(f"\n  {name}:")
    for k, v in m.items():
        if k != 'Strategy':
            print(f"    {k:20s}: {v}")

metrics_df = pd.DataFrame(metrics_rows)[
    ['Strategy','Ann_Return_%','Ann_Vol_%','Sharpe',
     'Max_Drawdown_%','CVaR_5pct_%','Calmar','N_Days']
]
metrics_df.to_csv('results/tables/portfolio_metrics.csv', index=False)
print("\n  Saved: results/tables/portfolio_metrics.csv")


print("\n" + "="*70)
print("CRISIS PERIOD PERFORMANCE")
print("="*70)

crisis_rows = []
for crisis_name, (start, end) in CRISIS_PERIODS.items():
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    mask = (common >= s) & (common <= e)
    if mask.sum() < 10:
        print(f"  {crisis_name}: insufficient data ({mask.sum()} days)")
        continue

    print(f"\n  {crisis_name} ({mask.sum()} days):")
    for name, r in [('Buy-and-Hold', ret_bnh),
                    ('VIX-Managed',  ret_vix),
                    ('CASSI-Managed', ret_cassi)]:
        m = compute_metrics(r[mask])
        m['Strategy'] = name
        m['Crisis']   = crisis_name
        crisis_rows.append(m)
        print(f"    {name:15s}  Return={m.get('Ann_Return_%','N/A'):6.1f}%  "
              f"MaxDD={m.get('Max_Drawdown_%','N/A'):6.1f}%  "
              f"Sharpe={m.get('Sharpe','N/A'):.2f}")

if crisis_rows:
    crisis_df = pd.DataFrame(crisis_rows)
    crisis_df.to_csv('results/tables/portfolio_crisis_metrics.csv', index=False)
    print("\n  Saved: results/tables/portfolio_crisis_metrics.csv")


print("\n" + "="*70)
print("MULTI-ASSET PORTFOLIO EXPERIMENT")
print("="*70)

all_assets   = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']
multi_rows   = []

for asset in all_assets:
    task = f'{asset}_q05_1w'
    if task not in pred_lgb_vix.columns or task not in pred_lgb_cassi.columns:
        continue
    if asset not in returns_df.columns:
        continue

    ret_asset    = returns_df[asset].dropna()
    vix_sig      = pred_lgb_vix[task].dropna()
    cassi_sig    = pred_lgb_cassi[task].dropna()
    common_a     = ret_asset.index.intersection(vix_sig.index).intersection(cassi_sig.index)

    if len(common_a) < 252:
        continue

    ret_a    = ret_asset.loc[common_a]
    ret_bnh_ = ret_a

    ret_vix_, _, _   = build_managed_portfolio(ret_a, vix_sig.reindex(common_a),   TRIGGER_PCT)
    ret_cas_, _, _   = build_managed_portfolio(ret_a, cassi_sig.reindex(common_a), TRIGGER_PCT)

    for name, r in [('BuyHold', ret_bnh_), ('VIX', ret_vix_), ('CASSI', ret_cas_)]:
        m = compute_metrics(r)
        m['Asset']    = asset
        m['Strategy'] = name
        multi_rows.append(m)

    sh_bnh  = compute_metrics(ret_bnh_).get('Sharpe', np.nan)
    sh_vix  = compute_metrics(ret_vix_).get('Sharpe', np.nan)
    sh_cas  = compute_metrics(ret_cas_).get('Sharpe', np.nan)
    print(f"  {asset:15s}  BuyHold Sharpe={sh_bnh:.2f}  "
          f"VIX Sharpe={sh_vix:.2f}  CASSI Sharpe={sh_cas:.2f}  "
          f"CASSI advantage={sh_cas - sh_vix:+.2f}")

if multi_rows:
    multi_df = pd.DataFrame(multi_rows)
    multi_df.to_csv('results/tables/portfolio_multi_asset.csv', index=False)
    print("\n  Saved: results/tables/portfolio_multi_asset.csv")


print("\n" + "="*70)
print("Generating portfolio performance figure...")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                          gridspec_kw={'height_ratios': [3, 1.5, 1]})
fig.suptitle('Portfolio Performance: Cross-Asset Tail Risk Signal (S&P 500, 1-Week Horizon)\n'
             'CASSI-managed vs VIX-managed vs Buy-and-Hold',
             fontsize=13, fontweight='bold')

colors = {'Buy-and-Hold': '#9E9E9E', 'VIX-Managed': '#FF9800', 'CASSI-Managed': '#F44336'}


ax = axes[0]
for name, r in [('Buy-and-Hold', ret_bnh),
                ('VIX-Managed',  ret_vix),
                ('CASSI-Managed', ret_cassi)]:
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum.values, label=name, color=colors[name],
            lw=1.8 if name != 'Buy-and-Hold' else 1.2,
            alpha=0.9)

for label, (s, e) in CRISIS_PERIODS.items():
    try:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.08, color='navy', zorder=0)
    except Exception:
        pass

ax.set_ylabel('Cumulative Wealth ($1 invested)')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.set_title(f'Trigger: bottom {TRIGGER_PCT*100:.0f}% of tail risk signal → '
             f'{REDUCED_EXPOSURE*100:.0f}% equity exposure', fontsize=9)
ax.set_xlim(common[0], common[-1])
ax.grid(alpha=0.3)


ax = axes[1]
for name, r in [('Buy-and-Hold', ret_bnh),
                ('VIX-Managed',  ret_vix),
                ('CASSI-Managed', ret_cassi)]:
    cum  = (1 + r).cumprod()
    roll = cum.cummax()
    dd   = (cum - roll) / roll * 100
    ax.fill_between(dd.index, dd.values, 0, alpha=0.35, color=colors[name], label=name)
    ax.plot(dd.index, dd.values, color=colors[name], lw=0.8, alpha=0.7)

for label, (s, e) in CRISIS_PERIODS.items():
    try:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color='navy', zorder=0)
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        ax.text(mid, ax.get_ylim()[0] * 1.1 if ax.get_ylim()[0] < 0 else -40,
                label.replace(' ', '\n'), ha='center', fontsize=7,
                color='navy', va='top')
    except Exception:
        pass

ax.set_ylabel('Drawdown (%)')
ax.set_xlim(common[0], common[-1])
ax.grid(alpha=0.3)


ax = axes[2]
ax.fill_between(exp_cassi.index, exp_cassi.values * 100, 100,
                where=exp_cassi.values < FULL_EXPOSURE,
                alpha=0.5, color='#F44336', label='Reduced exposure (CASSI signal)')
ax.plot(exp_cassi.index, exp_cassi.values * 100, color='#F44336', lw=0.6)
ax.axhline(100, color='grey', ls='--', lw=0.8)
ax.set_ylabel('CASSI\nExposure %')
ax.set_xlabel('Date')
ax.set_ylim(0, 110)
ax.set_xlim(common[0], common[-1])
ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/figures/fig_portfolio_performance.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("  Saved: results/figures/fig_portfolio_performance.png")


print("\n" + "="*70)
print("SUMMARY FOR PAPER (Economic Significance Section)")
print("="*70)

m_bnh  = compute_metrics(ret_bnh)
m_vix  = compute_metrics(ret_vix)
m_cas  = compute_metrics(ret_cassi)

print(f"\nS&P 500 managed portfolio (full period):")
print(f"  Buy-and-Hold  Sharpe={m_bnh['Sharpe']:.2f}  MaxDD={m_bnh['Max_Drawdown_%']:.1f}%  "
      f"Return={m_bnh['Ann_Return_%']:.1f}%")
print(f"  VIX-Managed   Sharpe={m_vix['Sharpe']:.2f}  MaxDD={m_vix['Max_Drawdown_%']:.1f}%  "
      f"Return={m_vix['Ann_Return_%']:.1f}%")
print(f"  CASSI-Managed Sharpe={m_cas['Sharpe']:.2f}  MaxDD={m_cas['Max_Drawdown_%']:.1f}%  "
      f"Return={m_cas['Ann_Return_%']:.1f}%")
print(f"\n  CASSI Sharpe advantage over VIX:  {m_cas['Sharpe'] - m_vix['Sharpe']:+.2f}")
print(f"  CASSI MaxDD improvement over BnH: {m_cas['Max_Drawdown_%'] - m_bnh['Max_Drawdown_%']:+.1f} pp")

print("\nDone. Files saved:")
for f in ['results/tables/portfolio_metrics.csv',
          'results/tables/portfolio_crisis_metrics.csv',
          'results/tables/portfolio_multi_asset.csv',
          'results/figures/fig_portfolio_performance.png']:
    print(f"  {f}")
