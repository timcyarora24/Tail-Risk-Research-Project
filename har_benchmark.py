
import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/predictions', exist_ok=True)
os.makedirs('results/tables',      exist_ok=True)
os.makedirs('results/figures',     exist_ok=True)


print("Loading data...")
returns_df = pd.read_csv('data/processed/returns.csv',
                          index_col=0, parse_dates=True)
returns_df.index = pd.to_datetime(returns_df.index).normalize()

def load_pred(fname):
    df = pd.read_csv(f'results/predictions/{fname}', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df

actual         = load_pred('actual_targets.csv')
pred_lgb_vix   = load_pred('vix_predictions.csv')
pred_lgb_cassi = load_pred('cassi_predictions.csv')


ALPHA          = 0.05
INITIAL_WINDOW = 252 * 3   # 3 years
RETRAIN_FREQ   = 22        # monthly retraining
LASSO_ALPHA    = 0.01

TARGET_ASSETS = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']
HORIZONS      = ['1w', '1m', '3m']
HORIZON_DAYS  = {'1w': 5, '1m': 22, '3m': 63}


print("Building HAR-RV features")

def build_har_features(returns_df, target_assets):
    
    har = pd.DataFrame(index=returns_df.index)

    for asset in target_assets:
        r = returns_df[asset]
        rv = r ** 2                         

        har[f'{asset}_rv_d']  = rv.shift(1)
        har[f'{asset}_rv_w']  = rv.shift(1).rolling(5).mean()
        har[f'{asset}_rv_m']  = rv.shift(1).rolling(22).mean()

        
        har[f'{asset}_ret_d'] = r.shift(1)
        har[f'{asset}_ret_w'] = r.shift(1).rolling(5).sum()

        
        har[f'{asset}_jump']  = np.maximum(
            0, har[f'{asset}_rv_d'] - har[f'{asset}_rv_w']
        )

        # Signed negative return (downside-specific)
        har[f'{asset}_neg_ret_d'] = np.minimum(r.shift(1), 0)
        har[f'{asset}_neg_rv_d']  = (np.minimum(r.shift(1), 0)) ** 2

    return har.dropna()

har_features = build_har_features(returns_df, TARGET_ASSETS)
print(f"HAR features: {har_features.shape[1]} cols x {har_features.shape[0]} obs")

# Build targets (same as feature_engineering.py)
targets = pd.DataFrame(index=returns_df.index)
for asset in TARGET_ASSETS:
    for hname, hdays in HORIZON_DAYS.items():
        targets[f'{asset}_q05_{hname}'] = (
            returns_df[asset].rolling(hdays).sum().shift(-hdays)
        )
targets = targets.dropna()

common = har_features.index.intersection(targets.index)
har_features = har_features.loc[common]
targets      = targets.loc[common]

print(f"Common dates: {common[0].date()} to {common[-1].date()} ({len(common)} obs)")

print("\n" + "="*65)
print("Training HAR-Quantile model (rolling walk-forward)")
print("="*65)

har_preds = {}

for asset in TARGET_ASSETS:
    asset_cols = [c for c in har_features.columns if c.startswith(asset + '_')]
    X_asset    = har_features[asset_cols]

    for horizon in HORIZONS:
        task = f'{asset}_q05_{horizon}'
        print(f"\n  Task: {task}")

        preds_out = []
        dates_out = []

        for i in range(INITIAL_WINDOW, len(common), RETRAIN_FREQ):
            te_end = min(i + RETRAIN_FREQ, len(common))

            y_tr_raw   = targets[task].iloc[:i]
            valid_mask = y_tr_raw.notna()
            y_tr       = y_tr_raw[valid_mask].values
            y_te       = targets[task].iloc[i:te_end]

            if len(y_tr) < 100 or len(y_te) == 0:
                continue

            X_tr = X_asset.iloc[:i][valid_mask].fillna(0).values
            X_te = X_asset.iloc[i:te_end].fillna(0).values

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            try:
                m = QuantileRegressor(quantile=ALPHA, alpha=LASSO_ALPHA,
                                      solver='highs', fit_intercept=True)
                m.fit(X_tr_s, y_tr)
                p = np.clip(m.predict(X_te_s), -0.99, 0.10)
            except Exception as e:
                print(f"    ERROR at step {i}: {e}")
                p = np.full(len(y_te), np.nan)

            preds_out.extend(p.tolist())
            dates_out.extend(y_te.index.tolist())

        har_preds[task] = pd.Series(preds_out, index=dates_out)
        print(f"    {len(preds_out)} predictions")

har_df = pd.DataFrame(har_preds)
har_df.to_csv('results/predictions/har_predictions.csv')
print(f"\nSaved: results/predictions/har_predictions.csv  {har_df.shape}")

print("\n" + "="*65)
print("EVALUATION: HAR vs LGB-VIX vs LGB-CASSI")
print("="*65)

def quantile_loss(y_true, y_pred, alpha=0.05):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1) * e)))

def ql_vec(y_true, y_pred, alpha=0.05):
    e = y_true - y_pred
    return np.where(e >= 0, alpha * e, (alpha - 1) * e)

def diebold_mariano(loss_bench, loss_new):
    d    = loss_bench - loss_new  
    n    = len(d)
    dbar = d.mean()
    
    lags = min(10, n // 5)
    gamma0 = np.var(d, ddof=1)
    hac_var = gamma0
    for lag in range(1, lags + 1):
        gamma_l = np.cov(d[lag:], d[:-lag])[0, 1]
        hac_var += 2 * (1 - lag / (lags + 1)) * gamma_l
    hac_var = max(hac_var, 1e-12)
    t_stat  = dbar / np.sqrt(hac_var / n)
    p_val   = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return t_stat, p_val

comparison_rows = []

for asset in TARGET_ASSETS:
    for horizon in HORIZONS:
        task = f'{asset}_q05_{horizon}'

        if task not in actual.columns or task not in har_preds:
            continue

        y = actual[task].dropna()

        preds = {}
        for name, df in [('HAR',       har_df),
                          ('LGB_VIX',   pred_lgb_vix),
                          ('LGB_CASSI', pred_lgb_cassi)]:
            if task in df.columns:
                preds[name] = df[task].reindex(y.index)

        
        valid = y.notna()
        for p in preds.values():
            valid = valid & p.notna()

        if valid.sum() < 30:
            continue

        y_t = y[valid].values
        p_t = {k: preds[k][valid].values for k in preds}

        ql   = {k: quantile_loss(y_t, p_t[k]) for k in p_t}
        viol = {k: (y_t < p_t[k]).mean() * 100 for k in p_t}

        
        dm_stats = {}
        for name in ['LGB_VIX', 'LGB_CASSI']:
            if name in p_t and 'HAR' in p_t:
                t, p = diebold_mariano(ql_vec(y_t, p_t['HAR']),
                                       ql_vec(y_t, p_t[name]))
                dm_stats[name] = (t, p)

        row = {
            'Asset'   : asset,
            'Horizon' : horizon,
            'N_Obs'   : int(valid.sum()),
        }
        for name in ['HAR', 'LGB_VIX', 'LGB_CASSI']:
            if name in ql:
                row[f'QL_{name}']   = round(ql[name],   6)
                row[f'Viol_{name}'] = round(viol[name], 2)

        if 'LGB_VIX' in dm_stats:
            row['DM_VIX_vs_HAR_p']   = round(dm_stats['LGB_VIX'][1],   4)
            row['DM_VIX_vs_HAR_sig']  = 'YES' if dm_stats['LGB_VIX'][1]   < 0.05 else 'no'
        if 'LGB_CASSI' in dm_stats:
            row['DM_CASSI_vs_HAR_p']  = round(dm_stats['LGB_CASSI'][1], 4)
            row['DM_CASSI_vs_HAR_sig'] = 'YES' if dm_stats['LGB_CASSI'][1] < 0.05 else 'no'

        if 'LGB_CASSI' in ql and 'HAR' in ql:
            row['CASSI_imp_over_HAR_%'] = round(
                (ql['HAR'] - ql['LGB_CASSI']) / abs(ql['HAR']) * 100, 2)

        comparison_rows.append(row)

        ql_har   = ql.get('HAR',       np.nan)
        ql_vix   = ql.get('LGB_VIX',   np.nan)
        ql_cassi = ql.get('LGB_CASSI', np.nan)
        dm_p_cas = dm_stats.get('LGB_CASSI', (np.nan, np.nan))[1]
        print(f"  {task:30s}  HAR={ql_har:.5f}  "
              f"VIX={ql_vix:.5f}  CASSI={ql_cassi:.5f}  "
              f"CASSI_vs_HAR_p={dm_p_cas:.3f}  "
              f"{'CASSI WINS' if ql_cassi < ql_har else 'HAR WINS':>10s}")

har_comp_df = pd.DataFrame(comparison_rows)
har_comp_df.to_csv('results/tables/har_comparison.csv', index=False)
print(f"\nSaved: results/tables/har_comparison.csv")

print("\n" + "="*65)
print("SUMMARY: Where does CASSI beat HAR?")
print("="*65)

if 'CASSI_imp_over_HAR_%' in har_comp_df.columns:
    wins  = (har_comp_df['CASSI_imp_over_HAR_%'] > 0).sum()
    total = har_comp_df['CASSI_imp_over_HAR_%'].notna().sum()
    sig   = (har_comp_df.get('DM_CASSI_vs_HAR_sig', pd.Series()) == 'YES').sum()
    avg   = har_comp_df['CASSI_imp_over_HAR_%'].mean()
    print(f"  CASSI beats HAR: {wins}/{total} tasks")
    print(f"  Significant (DM p<0.05): {sig}/{total}")
    print(f"  Average improvement: {avg:+.2f}%")

    by_horizon = har_comp_df.groupby('Horizon')['CASSI_imp_over_HAR_%'].mean()
    print(f"\n  By horizon:")
    for h, v in by_horizon.items():
        print(f"    {h}: {v:+.2f}%")

print("\nGenerating HAR comparison figure...")

if 'CASSI_imp_over_HAR_%' in har_comp_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('LGB-CASSI vs HAR Benchmark\n'
                 '% improvement in quantile loss (positive = CASSI better)',
                 fontsize=12, fontweight='bold')

    for ax_i, (model_col, title) in enumerate([
        ('CASSI_imp_over_HAR_%', 'LGB-CASSI vs HAR'),
    ]):
        ax = axes[ax_i]
        pivot = har_comp_df.pivot(index='Asset', columns='Horizon',
                                   values=model_col)[['1w', '1m', '3m']]
        # significance stars
        sig_col = 'DM_CASSI_vs_HAR_sig'
        sig_pivot = har_comp_df.pivot(index='Asset', columns='Horizon',
                                       values=sig_col)[['1w','1m','3m']] \
                    if sig_col in har_comp_df.columns else None

        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                       aspect='auto')
        plt.colorbar(im, ax=ax, label='% improvement')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['1W', '1M', '3M'])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([a.upper().replace('_', ' ') for a in pivot.index])
        ax.set_title(title)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isnan(val):
                    continue
                star = ''
                if sig_pivot is not None:
                    star = '*' if sig_pivot.values[i, j] == 'YES' else ''
                ax.text(j, i, f'{val:+.1f}{star}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if abs(val) > vmax * 0.6 else 'black')

    ax = axes[1]
    if 'Viol_HAR' in har_comp_df.columns and 'Viol_LGB_CASSI' in har_comp_df.columns:
        pivot_viol_har   = har_comp_df.pivot(index='Asset', columns='Horizon',
                                              values='Viol_HAR')[['1w','1m','3m']]
        pivot_viol_cassi = har_comp_df.pivot(index='Asset', columns='Horizon',
                                              values='Viol_LGB_CASSI')[['1w','1m','3m']]
        diff_viol = pivot_viol_cassi - pivot_viol_har
        im2 = ax.imshow(diff_viol.values, cmap='RdYlGn_r', vmin=-3, vmax=3,
                        aspect='auto')
        plt.colorbar(im2, ax=ax, label='Violation rate diff (pp)')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['1W', '1M', '3M'])
        ax.set_yticks(range(len(diff_viol.index)))
        ax.set_yticklabels([a.upper().replace('_', ' ') for a in diff_viol.index])
        ax.set_title('Violation Rate: CASSI − HAR (pp)\nNegative = CASSI closer to 5%')
        for i in range(diff_viol.shape[0]):
            for j in range(diff_viol.shape[1]):
                val = diff_viol.values[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if abs(val) > 2 else 'black')

    plt.tight_layout()
    plt.savefig('results/figures/fig_har_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/figures/fig_har_comparison.png")

print("\nDone. Files saved:")
for f in ['results/predictions/har_predictions.csv',
          'results/tables/har_comparison.csv',
          'results/figures/fig_har_comparison.png']:
    print(f"  {f}")

