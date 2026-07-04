
# model_training.py  —  5-MODEL STACK
# Models:
#   1. LASSO-AR        own-asset only, linear
#   2. LASSO-VIX       own-asset + VIX, linear
#   3. LASSO-CASSI     own-asset + cross-asset, linear
#   4. LGB-VIX         own-asset + VIX, nonlinear
#   5. LGB-CASSI       own-asset + cross-asset, nonlinear


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/predictions', exist_ok=True)
os.makedirs('results/tables',      exist_ok=True)


vix_features   = pd.read_csv('data/processed/vix_features.csv',
                              index_col=0, parse_dates=True)
cassi_features = pd.read_csv('data/processed/cassi_features.csv',
                              index_col=0, parse_dates=True)
targets        = pd.read_csv('data/processed/targets.csv',
                              index_col=0, parse_dates=True)
returns        = pd.read_csv('data/processed/returns.csv',
                              index_col=0, parse_dates=True)

for df in [vix_features, cassi_features, targets, returns]:
    df.index = pd.to_datetime(df.index).normalize()

common_dates = (vix_features.index
                .intersection(cassi_features.index)
                .intersection(targets.index)
                .intersection(returns.index))

vix_features   = vix_features.loc[common_dates]
cassi_features = cassi_features.loc[common_dates]
targets        = targets.loc[common_dates]
returns        = returns.loc[common_dates]

print(f"Date range    : {common_dates[0].date()} -> {common_dates[-1].date()}")
print(f"Observations  : {len(common_dates)}")
print(f"VIX features  : {vix_features.shape[1]}")
print(f"CASSI features: {cassi_features.shape[1]}")

vix_in_cassi = [c for c in cassi_features.columns if 'vix' in c.lower()]
print(f"VIX in CASSI  : {len(vix_in_cassi)} "
      f"{'CLEAN' if not vix_in_cassi else 'WARNING: ' + str(vix_in_cassi)}")

#  asset / horizon config 
TARGET_ASSETS = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']
HORIZONS      = ['1w', '1m', '3m']
ALPHA         = 0.05

def get_ar_features(asset):
    cols = [c for c in vix_features.columns if c.startswith(asset + '_')]
    return vix_features[cols]

print("\nAR features per asset:")
for a in TARGET_ASSETS:
    print(f"  {a:20s}  {get_ar_features(a).shape[1]} cols")

LGB_PARAMS = {
    'objective'        : 'quantile',
    'alpha'            : ALPHA,
    'metric'           : 'quantile',
    'num_leaves'       : 31,
    'learning_rate'    : 0.05,
    'min_child_samples': 20,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'verbose'          : -1,
    'n_jobs'           : -1,
}
NUM_BOOST_ROUND   = 300
EARLY_STOP_ROUNDS = 50
LASSO_ALPHA       = 0.01
INITIAL_WINDOW    = 252 * 3
RETRAIN_FREQ      = 22


def scale2(Xtr, Xte):
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)

def scale3(Xtr, Xval, Xte):
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xval), sc.transform(Xte)


def train_lasso(Xtr, ytr, Xte, tag, step):
    try:
        m = QuantileRegressor(quantile=ALPHA, alpha=LASSO_ALPHA,
                              solver='highs', fit_intercept=True)
        m.fit(Xtr, ytr)
        nz = int(np.sum(m.coef_ != 0))
        return np.clip(m.predict(Xte), -0.99, 0.10), nz
    except Exception as e:
        print(f"    LASSO {tag} step {step}: {e}")
        return np.full(Xte.shape[0], np.nan), 0


def train_lgb(Xtr, ytr, Xval, yval, Xte, tag, step):
    try:
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval   = lgb.Dataset(Xval, label=yval, reference=dtrain)
        m = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
        return np.clip(m.predict(Xte), -0.99, 0.10), m
    except Exception as e:
        print(f"    LGB {tag} step {step}: {e}")
        return np.full(Xte.shape[0], np.nan), None


MODEL_KEYS = ['lasso_ar', 'lasso_vix', 'lasso_cassi', 'lgb_vix', 'lgb_cassi']

all_preds   = {k: {} for k in MODEL_KEYS}
all_actuals = {}
rolling_imp = {}


print("\n" + "="*65)
print("Training 5 models:")
print("  LASSO-AR | LASSO-VIX | LASSO-CASSI | LGB-VIX | LGB-CASSI")
print("="*65)

for asset in TARGET_ASSETS:
    ar_feat = get_ar_features(asset)

    for horizon in HORIZONS:
        task = f'{asset}_q05_{horizon}'
        print(f"\nTask: {task}")

        preds     = {k: [] for k in MODEL_KEYS}
        actuals   = []
        dates_out = []
        n_steps   = 0
        step_imp  = []

        nz = {'lasso_ar': [], 'lasso_vix': [], 'lasso_cassi': []}

        for i in range(INITIAL_WINDOW, len(common_dates), RETRAIN_FREQ):

            te_end = min(i + RETRAIN_FREQ, len(common_dates))

            
            y_tr_raw   = targets[task].iloc[:i]
            valid_mask = y_tr_raw.notna()
            y_tr       = y_tr_raw[valid_mask]
            y_te       = targets[task].iloc[i:te_end]

            if len(y_tr) < 100 or len(y_te) == 0:
                continue

            
            Xar    = ar_feat.iloc[:i][valid_mask].fillna(0)
            Xvix   = vix_features.iloc[:i][valid_mask].fillna(0)
            Xcassi = cassi_features.iloc[:i][valid_mask].fillna(0)

            Xar_te    = ar_feat.iloc[i:te_end].fillna(0)
            Xvix_te   = vix_features.iloc[i:te_end].fillna(0)
            Xcassi_te = cassi_features.iloc[i:te_end].fillna(0)

           
            val_n  = max(int(len(y_tr) * 0.1), 22)
            y_val  = y_tr.iloc[-val_n:].values
            y_tr_  = y_tr.iloc[:-val_n].values

            
            ar_s,    ar_te_s    = scale2(Xar.iloc[:-val_n],    Xar_te)
            vix_s,   vix_te_s   = scale2(Xvix.iloc[:-val_n],   Xvix_te)
            cassi_s, cassi_te_s = scale2(Xcassi.iloc[:-val_n], Xcassi_te)

            ar_l, ar_v_l, ar_te_l      = scale3(Xar.iloc[:-val_n],
                                                 Xar.iloc[-val_n:], Xar_te)
            vix_l, vix_v_l, vix_te_l   = scale3(Xvix.iloc[:-val_n],
                                                 Xvix.iloc[-val_n:], Xvix_te)
            cas_l, cas_v_l, cassi_te_l = scale3(Xcassi.iloc[:-val_n],
                                                 Xcassi.iloc[-val_n:],
                                                 Xcassi_te)

            # MODEL 1 LASSO-AR 
            p_ar, nz_ar = train_lasso(ar_s, y_tr_, ar_te_s, 'AR', i)
            nz['lasso_ar'].append(nz_ar)

            # MODEL 2 LASSO-VIX 
            p_lv, nz_vix = train_lasso(vix_s, y_tr_, vix_te_s, 'VIX', i)
            nz['lasso_vix'].append(nz_vix)

            #  MODEL 3 LASSO-CASSI 
            p_lc, nz_cassi = train_lasso(cassi_s, y_tr_, cassi_te_s,
                                         'CASSI', i)
            nz['lasso_cassi'].append(nz_cassi)

            # MODEL 4 LGB-VIX
            p_lgv, _ = train_lgb(vix_l, y_tr_, vix_v_l, y_val,
                                 vix_te_l, 'LGB-VIX', i)

            # MODEL 5 LGB-CASSI
            p_lgc, model_cassi = train_lgb(cas_l, y_tr_, cas_v_l, y_val,
                                           cassi_te_l, 'LGB-CASSI', i)

           
            if model_cassi is not None:
                imp = model_cassi.feature_importance(importance_type='gain')
                step_imp.append(dict(zip(cassi_features.columns, imp)))

           
            preds['lasso_ar'].extend(p_ar.tolist())
            preds['lasso_vix'].extend(p_lv.tolist())
            preds['lasso_cassi'].extend(p_lc.tolist())
            preds['lgb_vix'].extend(p_lgv.tolist())
            preds['lgb_cassi'].extend(p_lgc.tolist())

            actuals.extend(y_te.values.tolist())
            dates_out.extend(y_te.index.tolist())
            n_steps += 1

        
        for k in MODEL_KEYS:
            all_preds[k][task] = pd.Series(preds[k], index=dates_out)
        all_actuals[task] = pd.Series(actuals, index=dates_out)

        
        if step_imp:
            avg_imp = (pd.DataFrame(step_imp)
                       .mean().sort_values(ascending=False))
            rolling_imp[task] = avg_imp
            out = avg_imp.reset_index()
            out.columns = ['feature', 'importance']
            out.to_csv(f'results/tables/rolling_importance_{task}.csv',
                       index=False)

        
        def _status(avg, total):
            if avg == 0:     return "WARNING: all zero — reduce LASSO_ALPHA"
            if avg == total: return "NOTE: no sparsity"
            return "OK"

        print(f"  {n_steps} steps | {len(preds['lasso_ar'])} preds")
        for key in ['lasso_ar', 'lasso_vix', 'lasso_cassi']:
            avg = np.mean(nz[key]) if nz[key] else 0
            tot = {'lasso_ar': ar_feat.shape[1],
                   'lasso_vix': vix_features.shape[1],
                   'lasso_cassi': cassi_features.shape[1]}[key]
            print(f"  {key:15s} nonzero: {avg:.1f}/{tot}"
                  f"  [{_status(avg, tot)}]")


FILE_MAP = {
    'lasso_ar'   : 'ar_predictions.csv',
    'lasso_vix'  : 'lasso_vix_predictions.csv',
    'lasso_cassi': 'lasso_cassi_predictions.csv',
    'lgb_vix'    : 'vix_predictions.csv',
    'lgb_cassi'  : 'cassi_predictions.csv',
}

print("\n" + "="*65)
print("Saving predictions ...")
for k, fname in FILE_MAP.items():
    df = pd.DataFrame(all_preds[k])
    df.to_csv(f'results/predictions/{fname}')
    print(f"  {fname:40s}  {df.shape}")

pd.DataFrame(all_actuals).to_csv('results/predictions/actual_targets.csv')


print("\n" + "="*65)
print("TOP 5 FEATURES (rolling in-sample, LGB-CASSI):")
print("="*65)
for task, imp in rolling_imp.items():
    print(f"\n{task}")
    for feat, val in imp.head(5).items():
        print(f"  {feat:35s}  {val:8.1f}")


print("\n" + "="*65)
ft  = 'sp500_q05_1w'
y_a = pd.DataFrame(all_actuals)[ft].dropna()
print(f"Sanity check — {ft}  (actual 5th pct = {y_a.quantile(0.05):.4f}):")
for k in MODEL_KEYS:
    s = pd.DataFrame(all_preds[k])[ft].reindex(y_a.index).dropna()
    if len(s) > 0:
        print(f"  {k:20s}  mean={s.mean():.4f}  std={s.std():.4f}")

print("\nNext step: python evaluation.py")