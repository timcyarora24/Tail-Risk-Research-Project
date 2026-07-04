

import pandas as pd
import numpy as np
import os

os.makedirs('data/processed', exist_ok=True)

returns = pd.read_csv('data/processed/returns.csv', index_col=0, parse_dates=True)
returns.index = pd.to_datetime(returns.index).normalize()

print(f"Returns shape  : {returns.shape}")
print(f"Columns        : {returns.columns.tolist()}")
print(f"Date range     : {returns.index[0].date()} -> {returns.index[-1].date()}")

# Dollar-index safety
if 'dollar_index' in returns.columns:
    dxy_valid = returns['dollar_index'].notna().mean()
    if dxy_valid < 0.3:
        print(f"WARNING: dollar_index sparse ({dxy_valid:.1%}) — using -eurusd proxy")
        returns['dollar_index'] = -returns['eurusd']
    else:
        print(f"dollar_index OK ({dxy_valid:.1%} non-NaN)")
else:
    print("WARNING: dollar_index missing — using -eurusd proxy")
    returns['dollar_index'] = -returns['eurusd']

def fast_avg_abs_corr(ret_df, window=20):
    cols = ret_df.columns.tolist()
    corr_series = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = ret_df[cols[i]].rolling(window).corr(ret_df[cols[j]])
            corr_series.append(c.abs())
    return pd.concat(corr_series, axis=1).mean(axis=1)

target_assets = ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread']

def build_own_asset_features(returns, target_assets):
    df = pd.DataFrame(index=returns.index)
    for asset in target_assets:
        ret_lag1 = returns[asset].shift(1)
        for lag in [1, 5, 22]:
            df[f'{asset}_ret_lag{lag}'] = returns[asset].shift(lag)
        for window in [20, 60]:
            df[f'{asset}_vol_{window}'] = ret_lag1.rolling(window).std()
        df[f'{asset}_mom_22'] = returns[asset].shift(1).rolling(22).sum()
    return df

own_asset_df = build_own_asset_features(returns, target_assets)


print("\n" + "="*60)
print("Building VIX-only features ...")

vix_features = own_asset_df.copy()
vix_lag = returns['vix'].shift(1)

vix_features['vix_level']       = vix_lag
vix_features['vix_change_1d']   = vix_lag.diff(1)
vix_features['vix_ma_20']       = vix_lag.rolling(20).mean()
vix_features['vix_std_20']      = vix_lag.rolling(20).std()
vix_features['vix_ma_60']       = vix_lag.rolling(60).mean()
vix_features['vix_ratio_20_60'] = (
    vix_features['vix_ma_20'] / (vix_features['vix_ma_60'] + 1e-8)
)

vix_features = vix_features.dropna()
n_own = own_asset_df.shape[1]
print(f"VIX features   : {vix_features.shape[1]} total "
      f"({n_own} own-asset + {vix_features.shape[1]-n_own} VIX)")
vix_features.to_csv('data/processed/vix_features.csv')
print("Saved vix_features.csv")


print("\n" + "="*60)
print("Building CASSI features (ZERO VIX — clean comparison) ...")

cassi_features = own_asset_df.copy()

# Bond market stress 
treasury_lag = returns['treasury'].shift(1)
cassi_features['treasury_vol_60']   = treasury_lag.rolling(60).std()
cassi_features['treasury_vol_20']   = treasury_lag.rolling(20).std()
cassi_features['yield_curve_slope'] = (
    treasury_lag.rolling(60).mean() - treasury_lag.rolling(5).mean()
)

# Credit market stress
credit_lag = returns['credit_spread'].shift(1)
cassi_features['credit_vol_60']     = credit_lag.rolling(60).std()
cassi_features['credit_vol_20']     = credit_lag.rolling(20).std()

# Currency market stress
dxy_lag = returns['dollar_index'].shift(1)
eur_lag = returns['eurusd'].shift(1)
jpy_lag = returns['jpyusd'].shift(1) if 'jpyusd' in returns.columns else -eur_lag

cassi_features['dollar_vol_20']     = dxy_lag.rolling(20).std()
cassi_features['eurusd_vol_60']     = eur_lag.rolling(60).std()
cassi_features['eurusd_vol_20']     = eur_lag.rolling(20).std()
cassi_features['jpyusd_vol_20']     = jpy_lag.rolling(20).std()
cassi_features['safe_haven_jpy']    = jpy_lag.rolling(5).sum()

# Commodity market stress
oil_lag    = returns['oil'].shift(1)
gold_lag   = (returns['gold'].shift(1) if 'gold' in returns.columns
              else pd.Series(0, index=returns.index))
copper_lag = (returns['copper'].shift(1) if 'copper' in returns.columns
              else pd.Series(0, index=returns.index))

cassi_features['gold_vol_20']       = gold_lag.rolling(20).std()
cassi_features['oil_vol_60']        = oil_lag.rolling(60).std()
cassi_features['gold_copper_ratio'] = (
    gold_lag.rolling(20).mean() / (copper_lag.rolling(20).mean().abs() + 1e-8)
)

#  Cross-asset correlations
ret_lag = returns.shift(1)


cassi_features['stock_bond_corr']     = (
    ret_lag['sp500'].rolling(20).corr(ret_lag['treasury'])
)

cassi_features['corr_dollar_jpyusd']  = (
    ret_lag['dollar_index'].rolling(20).corr(ret_lag['jpyusd'])
    if 'jpyusd' in returns.columns else pd.Series(0, index=returns.index)
)

cassi_features['corr_oil_jpyusd']     = (
    ret_lag['oil'].rolling(20).corr(ret_lag['jpyusd'])
    if 'jpyusd' in returns.columns else pd.Series(0, index=returns.index)
)

cassi_features['corr_treasury_copper'] = (
    ret_lag['treasury'].rolling(20).corr(ret_lag['copper'])
    if 'copper' in returns.columns else pd.Series(0, index=returns.index)
)

cassi_features['corr_sp500_dollar']   = (
    ret_lag['sp500'].rolling(20).corr(ret_lag['dollar_index'])
)

cassi_features['corr_eurusd_copper']  = (
    ret_lag['eurusd'].rolling(20).corr(ret_lag['copper'])
    if 'copper' in returns.columns else pd.Series(0, index=returns.index)
)

cassi_features['corr_credit_oil']     = (
    ret_lag['credit_spread'].rolling(20).corr(ret_lag['oil'])
)


all_sync_assets = [c for c in
    ['sp500', 'treasury', 'eurusd', 'oil', 'credit_spread',
     'dollar_index', 'gold', 'copper']
    if c in returns.columns]

cassi_features['avg_abs_corr_20'] = fast_avg_abs_corr(
    returns[all_sync_assets].shift(1), window=20
)


vix_leak = [c for c in cassi_features.columns if 'vix' in c.lower()]
if vix_leak:
    print(f"  WARNING removing leaked VIX cols: {vix_leak}")
    cassi_features = cassi_features.drop(columns=vix_leak)
else:
    print("  Confirmed: ZERO VIX features in CASSI")

cassi_features = cassi_features.dropna()

own_cols  = [c for c in cassi_features.columns
             if any(c.startswith(a + '_') for a in target_assets)]
cross_cols = [c for c in cassi_features.columns if c not in own_cols]

print(f"CASSI features : {cassi_features.shape[1]} total x {cassi_features.shape[0]} obs")
print(f"  Own-asset    : {len(own_cols)}")
print(f"  Cross-asset  : {len(cross_cols)} (no VIX)")
print(f"  Cross cols   : {cross_cols}")
cassi_features.to_csv('data/processed/cassi_features.csv')
print("Saved cassi_features.csv")


print("\n" + "="*60)
print("Building target variables ...")

horizons = {'1w': 5, '1m': 22, '3m': 63}
targets  = pd.DataFrame(index=returns.index)

for asset in target_assets:
    for hname, hdays in horizons.items():
        targets[f'{asset}_q05_{hname}'] = (
            returns[asset].rolling(hdays).sum().shift(-hdays)
        )

targets = targets.dropna()
print(f"Targets : {targets.shape[1]} cols x {targets.shape[0]} obs")
print(f"Range   : {targets.index[0].date()} -> {targets.index[-1].date()}")
print("\n5th-percentile check:")
for col in targets.columns[:6]:
    print(f"  {col:30s} q05={targets[col].quantile(0.05):.4f}  "
          f"med={targets[col].quantile(0.50):.4f}  "
          f"q95={targets[col].quantile(0.95):.4f}")

targets.to_csv('data/processed/targets.csv')
print("\nSaved targets.csv")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"VIX features  : {vix_features.shape[1]} ({n_own} own + "
      f"{vix_features.shape[1]-n_own} VIX)")
print(f"CASSI features: {cassi_features.shape[1]} ({len(own_cols)} own + "
      f"{len(cross_cols)} cross-asset, 0 VIX)")
print(f"Targets       : {targets.shape[1]} targets, {targets.shape[0]} obs")
print(f"\nClean comparison established:")
print(f"  VIX   model = own-asset + VIX signals only")
print(f"  CASSI model = own-asset + cross-asset (no VIX)")