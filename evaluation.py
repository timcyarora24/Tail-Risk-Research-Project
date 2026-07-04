
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/tables', exist_ok=True)

try:
    from arch.bootstrap import MCS
    MCS_AVAILABLE = True
    print("arch library found — MCS test will run")
except ImportError:
    MCS_AVAILABLE = False
    print("arch library not found — skipping MCS")
    print("  Install with: pip install arch")


def load_pred(fname):
    df = pd.read_csv(f'results/predictions/{fname}',
                     index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df

pred_ar          = load_pred('ar_predictions.csv')
pred_lasso_vix   = load_pred('lasso_vix_predictions.csv')
pred_lasso_cassi = load_pred('lasso_cassi_predictions.csv')
pred_lgb_vix     = load_pred('vix_predictions.csv')
pred_lgb_cassi   = load_pred('cassi_predictions.csv')
actual           = load_pred('actual_targets.csv')

models = {
    'LASSO_AR'   : pred_ar,
    'LASSO_VIX'  : pred_lasso_vix,
    'LASSO_CASSI': pred_lasso_cassi,
    'LGB_VIX'    : pred_lgb_vix,
    'LGB_CASSI'  : pred_lgb_cassi,
}

print("\nPredictions loaded:")
for name, df in models.items():
    print(f"  {name:15s}  {df.shape}")


vix_raw = pd.read_csv('data/raw/vix.csv', header=0)
date_col = vix_raw.columns[0]
junk = {date_col, 'Ticker', 'Price', 'Date', 'date'}
vix_raw = vix_raw[~vix_raw[date_col].isin(junk)]
vix_raw[date_col] = pd.to_datetime(vix_raw[date_col], errors='coerce')
vix_raw = vix_raw.dropna(subset=[date_col]).set_index(date_col)
vix_raw.index = vix_raw.index.normalize()

if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = ['_'.join([str(c) for c in col if c]).strip('_')
                       for col in vix_raw.columns]

close_cands = [c for c in vix_raw.columns if 'close' in c.lower()]
vix_close = pd.to_numeric(
    vix_raw[close_cands[0]] if close_cands else vix_raw.iloc[:, 0],
    errors='coerce'
).dropna()

normal_idx      = vix_close[vix_close <= 20].index.normalize()
stress_idx      = vix_close[vix_close > 20].index.normalize()
high_stress_idx = vix_close[vix_close > 30].index.normalize()

print(f"\nVIX regimes:")
print(f"  Normal      VIX<=20 : {len(normal_idx):5d} days")
print(f"  Stress      VIX>20  : {len(stress_idx):5d} days")
print(f"  High Stress VIX>30  : {len(high_stress_idx):5d} days")


ALPHA = 0.05

def quantile_loss(y_true, y_pred, alpha=ALPHA):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha-1) * e)))

def ql_vec(y_true, y_pred, alpha=ALPHA):
    e = y_true - y_pred
    return np.where(e >= 0, alpha * e, (alpha-1) * e)

def diebold_mariano(loss_bench, loss_new):
    
    d = loss_bench - loss_new
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    d_mean  = np.mean(d)
    bw      = int(np.ceil(n ** (1/3)))
    hac_var = np.var(d, ddof=1)
    for h in range(1, bw + 1):
        gamma_h = np.mean((d[h:] - d_mean) * (d[:-h] - d_mean))
        hac_var += 2 * (1 - h/(bw+1)) * gamma_h
    hac_var = max(hac_var, 1e-20)
    t_stat  = d_mean / np.sqrt(hac_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return float(t_stat), float(p_value)

def kupiec_pof_test(y_true, y_pred_q, alpha=ALPHA):
    
    viol   = (y_true < y_pred_q).astype(float)
    n      = len(viol)
    n_v    = viol.sum()
    vr     = n_v / n
    if n_v == 0 or n_v == n:
        return vr, np.nan, np.nan, 'n/a'
    p_hat = vr
    try:
        ll_null = n_v*np.log(alpha) + (n-n_v)*np.log(1-alpha)
        ll_alt  = n_v*np.log(p_hat) + (n-n_v)*np.log(1-p_hat)
        lr      = -2*(ll_null - ll_alt)
        pv      = 1 - chi2.cdf(lr, df=1)
    except Exception:
        return vr, np.nan, np.nan, 'error'
    return float(vr), float(lr), float(pv), 'PASS' if pv >= 0.05 else 'FAIL'

def christoffersen_test(y_true, y_pred_q, alpha=ALPHA):
    
    viol = (y_true < y_pred_q).astype(int)
    n    = len(viol)

   
    n00 = sum(1 for i in range(n-1) if viol[i]==0 and viol[i+1]==0)
    n01 = sum(1 for i in range(n-1) if viol[i]==0 and viol[i+1]==1)
    n10 = sum(1 for i in range(n-1) if viol[i]==1 and viol[i+1]==0)
    n11 = sum(1 for i in range(n-1) if viol[i]==1 and viol[i+1]==1)

    denom01 = n00 + n01
    denom11 = n10 + n11
    denom   = n00 + n01 + n10 + n11

    if denom01 == 0 or denom11 == 0 or denom == 0:
        return np.nan, np.nan, 'n/a'

    pi01 = n01 / denom01
    pi11 = n11 / denom11 if denom11 > 0 else 0
    pi   = (n01 + n11) / denom

    if pi in (0, 1) or pi01 in (0, 1):
        return np.nan, np.nan, 'n/a'

    try:
        L_ind = ((1-pi)**(n00+n10)) * (pi**(n01+n11))
        L_dep = (((1-pi01)**n00) * (pi01**n01) *
                 ((1-pi11)**n10) * (pi11**n11 if n11 > 0 else 1))
        if L_ind <= 0 or L_dep <= 0:
            return np.nan, np.nan, 'n/a'
        lr  = -2 * np.log(L_ind / L_dep)
        pv  = 1 - chi2.cdf(lr, df=1)
    except Exception:
        return np.nan, np.nan, 'error'

    return float(lr), float(pv), 'PASS' if pv >= 0.05 else 'FAIL'

def run_mcs(loss_dict, size=0.10):
    
    if not MCS_AVAILABLE:
        return None
    try:
        loss_df = pd.DataFrame(loss_dict)
        mcs     = MCS(loss_df, size=size, method='max')
        mcs.compute()
        return list(mcs.included)
    except Exception as e:
        return None


target_assets = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']
horizons      = ['1w', '1m', '3m']

main_results    = []
kupiec_results  = []
christ_results  = []
crisis_results  = []
mcs_results     = []

print("\n" + "="*80)
print("EVALUATION: LASSO-AR | LASSO-VIX | LASSO-CASSI | LGB-VIX | LGB-CASSI")
print("="*80)

for target_asset in target_assets:
    for horizon in horizons:
        task = f'{target_asset}_q05_{horizon}'

        if task not in actual.columns:
            continue

        y_true = actual[task].dropna()

        
        preds_dict = {}
        for mname, mdf in models.items():
            if task in mdf.columns:
                preds_dict[mname] = mdf[task].reindex(y_true.index)

        valid = y_true.notna()
        for p in preds_dict.values():
            valid = valid & p.notna()

        y_t = y_true[valid].values
        if len(y_t) < 30:
            continue

        y_preds = {k: preds_dict[k][valid].values for k in preds_dict}

        
        ql = {k: quantile_loss(y_t, y_preds[k]) for k in y_preds}

        
        def pct_imp(b, n):
            bv, nv = ql.get(b, np.nan), ql.get(n, np.nan)
            if np.isnan(bv) or bv == 0: return np.nan
            return (bv - nv) / abs(bv) * 100

        
        def dm(b, n):
            if b not in y_preds or n not in y_preds:
                return np.nan, np.nan
            return diebold_mariano(ql_vec(y_t, y_preds[b]),
                                   ql_vec(y_t, y_preds[n]))

        dm_vix_ar_t,     dm_vix_ar_p     = dm('LASSO_AR',    'LGB_VIX')
        dm_lasso_comp_t, dm_lasso_comp_p = dm('LASSO_VIX',   'LASSO_CASSI')
        dm_cassi_vix_t,  dm_cassi_vix_p  = dm('LGB_VIX',     'LGB_CASSI')
        dm_nlr_t,        dm_nlr_p        = dm('LASSO_VIX',   'LGB_VIX')

        
        for mname in y_preds:

            # Kupiec
            vr, lr_k, pv_k, res_k = kupiec_pof_test(y_t, y_preds[mname])
            kupiec_results.append({
                'Asset': target_asset, 'Horizon': horizon, 'Model': mname,
                'Violation_Rate_%': round(vr*100, 2),
                'Expected_%'      : 5.0,
                'LR_Stat'         : round(lr_k, 3) if not np.isnan(lr_k) else np.nan,
                'p_value'         : round(pv_k, 4) if not np.isnan(pv_k) else np.nan,
                'Kupiec'          : res_k,
                'N_Obs'           : int(valid.sum()),
            })

            
            lr_c, pv_c, res_c = christoffersen_test(y_t, y_preds[mname])
            christ_results.append({
                'Asset'    : target_asset,
                'Horizon'  : horizon,
                'Model'    : mname,
                'LR_Stat'  : round(lr_c, 3) if not np.isnan(lr_c) else np.nan,
                'p_value'  : round(pv_c, 4) if not np.isnan(pv_c) else np.nan,
                'CC_Test'  : res_c,
                'N_Obs'    : int(valid.sum()),
            })

        
        if MCS_AVAILABLE:
            loss_dict = {k: ql_vec(y_t, y_preds[k]) for k in y_preds}
            mcs_included = run_mcs(loss_dict, size=0.10)
            if mcs_included is not None:
                for mname in y_preds:
                    mcs_results.append({
                        'Asset'      : target_asset,
                        'Horizon'    : horizon,
                        'Model'      : mname,
                        'In_MCS_90%' : 'YES' if mname in mcs_included else 'no',
                    })

        
        row = {
            'Asset': target_asset, 'Horizon': horizon,
            'N_Obs': int(valid.sum()),
        }
        for mname in y_preds:
            row[f'QL_{mname}'] = round(ql[mname], 6)

        row['Imp_VIX_over_AR_%']          = round(pct_imp('LASSO_AR',  'LGB_VIX'),    2)
        row['Imp_LASSO_CASSI_over_VIX_%'] = round(pct_imp('LASSO_VIX', 'LASSO_CASSI'),2)
        row['Imp_CASSI_over_VIX_%']       = round(pct_imp('LGB_VIX',   'LGB_CASSI'),  2)
        row['Imp_LGB_over_LASSO_%']       = round(pct_imp('LASSO_VIX', 'LGB_VIX'),    2)

        row['DM_VIX_vs_AR_p']       = round(dm_vix_ar_p,     4) if not np.isnan(dm_vix_ar_p)     else np.nan
        row['DM_LASSO_CASSI_VIX_p'] = round(dm_lasso_comp_p, 4) if not np.isnan(dm_lasso_comp_p) else np.nan
        row['DM_CASSI_vs_VIX_p']    = round(dm_cassi_vix_p,  4) if not np.isnan(dm_cassi_vix_p)  else np.nan
        row['DM_NLR_p']             = round(dm_nlr_p,         4) if not np.isnan(dm_nlr_p)        else np.nan

        row['CASSI_sig_5pct'] = ('YES' if (not np.isnan(dm_cassi_vix_p)
                                           and dm_cassi_vix_p < 0.05) else 'no')
        main_results.append(row)

        
        valid_index = y_true[valid].index
        regimes = {
            'Normal'       : ~valid_index.isin(stress_idx),
            'Stress_VIX20' : valid_index.isin(stress_idx),
            'HighStr_VIX30': valid_index.isin(high_stress_idx),
        }

        for regime, mask_r in regimes.items():
            n_r = int(mask_r.sum())
            if n_r < 10:
                continue
            ql_v = quantile_loss(y_t[mask_r], y_preds['LGB_VIX'][mask_r])
            ql_c = quantile_loss(y_t[mask_r], y_preds['LGB_CASSI'][mask_r])
            imp  = (ql_v - ql_c) / abs(ql_v) * 100 if ql_v != 0 else 0
            dm_t_r, dm_p_r = diebold_mariano(
                ql_vec(y_t[mask_r], y_preds['LGB_VIX'][mask_r]),
                ql_vec(y_t[mask_r], y_preds['LGB_CASSI'][mask_r])
            )
            crisis_results.append({
                'Asset': target_asset, 'Horizon': horizon, 'Regime': regime,
                'QL_LGB_VIX'  : round(ql_v, 6),
                'QL_LGB_CASSI': round(ql_c, 6),
                'CASSI_Imp_%' : round(imp, 2),
                'DM_p'        : round(dm_p_r, 4) if not np.isnan(dm_p_r) else np.nan,
                'Sig_5pct'    : 'YES' if (not np.isnan(dm_p_r) and dm_p_r < 0.05) else 'no',
                'N_Obs'       : n_r,
            })

        print(f"  {task:35s}  "
              f"CASSI/VIX={row.get('Imp_CASSI_over_VIX_%', np.nan):+5.1f}%  "
              f"DM_p={dm_cassi_vix_p:.3f}")


results_df  = pd.DataFrame(main_results)
kupiec_df   = pd.DataFrame(kupiec_results)
christ_df   = pd.DataFrame(christ_results)
crisis_df   = pd.DataFrame(crisis_results)

results_df.to_csv('results/tables/comparison_metrics.csv', index=False)
kupiec_df.to_csv('results/tables/kupiec_results.csv',      index=False)
christ_df.to_csv('results/tables/christoffersen_results.csv', index=False)
crisis_df.to_csv('results/tables/crisis_analysis.csv',     index=False)

if mcs_results:
    mcs_df = pd.DataFrame(mcs_results)
    mcs_df.to_csv('results/tables/mcs_results.csv', index=False)


print("\n" + "="*80)
print("TABLE 1: QUANTILE LOSS — ALL 5 MODELS (lower = better)")
print("="*80)
ql_cols = [c for c in results_df.columns if c.startswith('QL_')]
for asset in target_assets:
    sub = results_df[results_df['Asset']==asset][['Horizon']+ql_cols]
    print(f"\n{asset.upper()}")
    print(sub.to_string(index=False))

print("\n" + "="*80)
print("TABLE 2: IMPROVEMENT CHAIN")
print("  Q1: LGB-VIX   over LASSO-AR    — VIX adds value?")
print("  Q2: LASSO-CASSI over LASSO-VIX — CASSI linear?")
print("  Q3: LGB-CASSI over LGB-VIX     — CASSI nonlinear?")
print("  Q4: LGB-VIX   over LASSO-VIX   — nonlinearity helps?")
print("="*80)
imp_cols = ['Asset','Horizon',
            'Imp_VIX_over_AR_%','Imp_LASSO_CASSI_over_VIX_%',
            'Imp_CASSI_over_VIX_%','Imp_LGB_over_LASSO_%']
print(results_df[imp_cols].to_string(index=False))

print("\n" + "="*80)
print("TABLE 3: DM TEST p-VALUES (4 key comparisons)")
print("="*80)
dm_cols = ['Asset','Horizon',
           'DM_VIX_vs_AR_p','DM_LASSO_CASSI_VIX_p',
           'DM_CASSI_vs_VIX_p','DM_NLR_p']
print(results_df[dm_cols].to_string(index=False))

print("\n" + "="*80)
print("TABLE 4: KUPIEC POF TEST (violation frequency calibration)")
print("  PASS = violation rate not significantly different from 5%")
print("="*80)
kup_col = 'Kupiec' if 'Kupiec' in kupiec_df.columns else 'Calibration'
kup_pivot = kupiec_df.pivot_table(
    index=['Asset','Model'], columns='Horizon',
    values=kup_col, aggfunc='first')[['1w','1m','3m']]
print(kup_pivot.to_string())

print("\n--- Violation Rates (%) ---")
vr_pivot = kupiec_df.pivot_table(
    index=['Asset','Model'], columns='Horizon',
    values='Violation_Rate_%', aggfunc='first')[['1w','1m','3m']]
print(vr_pivot.to_string())

print("\n" + "="*80)
print("TABLE 5: CHRISTOFFERSEN INDEPENDENCE TEST (violation clustering)")
print("  PASS = violations are not clustered (good risk model)")
print("  FAIL = violations cluster in time (bad: misses crisis periods)")
print("="*80)
cc_pivot = pd.DataFrame(christ_results).pivot_table(
    index=['Asset','Model'], columns='Horizon',
    values='CC_Test', aggfunc='first')[['1w','1m','3m']]
print(cc_pivot.to_string())

print("\n" + "="*80)
print("TABLE 6: CRISIS ANALYSIS — CASSI vs VIX (LGB models)")
print("="*80)
if len(crisis_df) > 0:
    sub = crisis_df[crisis_df['Regime'].isin(['Normal','Stress_VIX20'])]
    cpivot = sub.pivot_table(
        index=['Asset','Regime'], columns='Horizon',
        values='CASSI_Imp_%', aggfunc='first')[['1w','1m','3m']]
    print(cpivot.to_string())
    n_avg = crisis_df[crisis_df['Regime']=='Normal']['CASSI_Imp_%'].mean()
    s_avg = crisis_df[crisis_df['Regime']=='Stress_VIX20']['CASSI_Imp_%'].mean()
    h_avg = crisis_df[crisis_df['Regime']=='HighStr_VIX30']['CASSI_Imp_%'].mean()
    print(f"\nAverage CASSI improvement by regime:")
    print(f"  Normal      (VIX<=20): {n_avg:+.2f}%")
    print(f"  Stress      (VIX>20) : {s_avg:+.2f}%")
    print(f"  High Stress (VIX>30) : {h_avg:+.2f}%")
    print(f"  Hypothesis supported: {'YES' if s_avg > n_avg else 'NO'}")

if mcs_results:
    print("\n" + "="*80)
    print("TABLE 7: MODEL CONFIDENCE SET (90% MCS)")
    print("  YES = model in MCS (cannot be statistically excluded)")
    print("="*80)
    mcs_df = pd.DataFrame(mcs_results)
    mcs_pivot = mcs_df.pivot_table(
        index=['Asset','Model'], columns='Horizon',
        values='In_MCS_90%', aggfunc='first')[['1w','1m','3m']]
    print(mcs_pivot.to_string())

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
n_wins  = (results_df['Imp_CASSI_over_VIX_%'] > 0).sum()
n_sig   = (results_df['CASSI_sig_5pct'] == 'YES').sum()
avg_imp = results_df['Imp_CASSI_over_VIX_%'].mean()

lgb_kup = kupiec_df[kupiec_df['Model']=='LGB_CASSI']
n_kup_pass = (lgb_kup['Kupiec'] == 'PASS').sum()

lgb_cc  = pd.DataFrame(christ_results)
lgb_cc  = lgb_cc[lgb_cc['Model']=='LGB_CASSI']
n_cc_pass = (lgb_cc['CC_Test'] == 'PASS').sum()

print(f"\nLGB-CASSI vs LGB-VIX:")
print(f"  Avg improvement          : {avg_imp:+.2f}%")
print(f"  Tasks where CASSI wins   : {n_wins}/15")
print(f"  Statistically sig (DM)   : {n_sig}/15")
print(f"\nLGB-CASSI calibration:")
print(f"  Kupiec PASS              : {n_kup_pass}/15")
print(f"  Christoffersen PASS      : {n_cc_pass}/15")

print(f"\nSaved to results/tables/:")
for f in ['comparison_metrics.csv','kupiec_results.csv',
          'christoffersen_results.csv','crisis_analysis.csv',
          'mcs_results.csv (if arch installed)']:
    print(f"  {f}")