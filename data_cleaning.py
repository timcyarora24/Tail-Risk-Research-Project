
import pandas as pd
import numpy as np
import os

os.makedirs('data/processed', exist_ok=True)


ASSETS = {
    'sp500'        : {'path': 'data/raw/sp500.csv',         'type': 'price'},
    'treasury'     : {'path': 'data/raw/treasury.csv',      'type': 'price'},
    'eurusd'       : {'path': 'data/raw/eurusd.csv',        'type': 'price'},
    'oil'          : {'path': 'data/raw/crude_oil.csv',     'type': 'price'},
    'vix'          : {'path': 'data/raw/vix.csv',           'type': 'price'},
    'credit_spread': {'path': 'data/raw/credit_spread.csv', 'type': 'price'},
    'dollar_index' : {'path': 'data/raw/dollar_index.csv',  'type': 'price'},
    'gold'         : {'path': 'data/raw/gold.csv',          'type': 'price'},
    'copper'       : {'path': 'data/raw/copper.csv',        'type': 'price'},
    'jpyusd'       : {'path': 'data/raw/jpyusd.csv',        'type': 'price'},
}


def load_series(name, path):
    """
    Loads a yfinance CSV and returns a clean pd.Series of prices.
    Handles yfinance's multi-row header (Ticker/Price rows on rows 0-1).
    Tries Adj Close -> Close -> first numeric column in that order.
    """
    try:
        df = pd.read_csv(path)

        if df.empty or df.shape[1] < 2:
            return None, f"empty file"

        
        df.index = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df[df.index.notna()]
        df = df.iloc[:, 1:]           # drop the date column
        df.index = df.index.normalize()

        
        junk = {'ticker', 'price', 'date', 'adj close', 'close'}
        for col in df.columns:
            mask = df[col].astype(str).str.strip().str.lower().isin(junk)
            df = df[~mask]

        
        col_map = {c.strip().lower(): c for c in df.columns}
        if 'adj close' in col_map:
            s = df[col_map['adj close']]
        elif 'close' in col_map:
            s = df[col_map['close']]
        else:
            s = df.iloc[:, 0]

        s = pd.to_numeric(s, errors='coerce')
        s.index = df.index
        s.name  = name
        s = s.dropna()
        s = s[~s.index.duplicated(keep='last')]
        s = s.sort_index()

        return s, None

    except Exception as e:
        return None, str(e)


print("Loading raw data ...")
print("-" * 65)

raw = {}
for name, cfg in ASSETS.items():
    s, err = load_series(name, cfg['path'])
    if s is not None and len(s) > 50:
        raw[name] = s
        print(f"  {name:20s}  {len(s):5d} obs  "
              f"{s.index.min().date()} -> {s.index.max().date()}  "
              f"[{s.min():.4f}, {s.max():.4f}]")
    else:
        print(f"  {name:20s}  SKIPPED  {err or 'too few rows'}")


if 'dollar_index' not in raw and 'eurusd' in raw:
    print("  dollar_index missing — using -EURUSD as proxy")
    raw['dollar_index'] = -raw['eurusd'].rename('dollar_index')

print(f"\nLoaded: {len(raw)}/10 assets")


if 'oil' in raw:
    n_neg = (raw['oil'] <= 0).sum()
    if n_neg > 0:
        print(f"\nOil: {n_neg} non-positive prices detected "
              f"(April 2020 WTI event) — clipping to floor of $0.01")
        raw['oil'] = raw['oil'].clip(lower=0.01)


print("\nAligning to common date range ...")
min_date = max(s.index.min() for s in raw.values())
max_date = min(s.index.max() for s in raw.values())
print(f"  Range: {min_date.date()} -> {max_date.date()}")

bdays = pd.bdate_range(min_date, max_date)
for name in raw:
    raw[name] = raw[name].reindex(bdays)   


nan_counts = {n: int(raw[n].isna().sum()) for n in raw if raw[n].isna().sum() > 0}
if nan_counts:
    print("\n  NaN gaps (will be dropped):")
    for n, c in nan_counts.items():
        print(f"    {n:20s}  {c} rows")


print("\nComputing returns ...")
ret = {}
for name, cfg in ASSETS.items():
    if name not in raw:
        continue
    s = raw[name]
    if cfg['type'] == 'spread':
        r = s.diff()
        label = 'first-diff'
    else:
        r = np.log(s / s.shift(1))
        label = 'log-return'
    ret[name] = r
    print(f"  {name:20s}  {label}  "
          f"mean={r.mean():.5f}  std={r.std():.5f}")

returns = pd.DataFrame(ret).dropna()   

print(f"\nReturns shape : {returns.shape}")
print(f"Date range    : {returns.index[0].date()} -> {returns.index[-1].date()}")
print(f"Trading days  : {len(returns)}")
print(f"Years of data : {len(returns)/252:.1f}")


print("\n=== SANITY CHECKS ===")

missing = returns.isnull().sum()
inf_cnt = np.isinf(returns).sum()
print(f"NaN  (all should be 0): {missing.sum()}")
print(f"Inf  (all should be 0): {inf_cnt.sum()}")

if missing.sum() > 0 or inf_cnt.sum() > 0:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Cleaned. New shape: {returns.shape}")

print("\nSummary statistics:")
print(returns.describe().round(5).to_string())

print("\nCorrelation matrix:")
print(returns.corr().round(3).to_string())


print("\n=== CRISIS VERIFICATION ===")
crises = {
    '2008 GFC peak'  : ('2008-09-01', '2008-12-31'),
    '2020 COVID crash': ('2020-02-01', '2020-04-30'),
    '2022 rate shock' : ('2022-01-01', '2022-12-31'),
}
for label, (s, e) in crises.items():
    window = returns['sp500'].loc[s:e]
    if len(window) > 0:
        print(f"  {label:25s}  sp500 min={window.min():.4f}  "
              f"on {window.idxmin().date()}")


if 'credit_spread' in returns.columns:
    hyg = returns['credit_spread']
    print(f"\nHYG (credit spread proxy):")
    print(f"  Worst day : {hyg.min():.5f}  on {hyg.idxmin().date()}")
    print(f"  Best day  : {hyg.max():.5f}  on {hyg.idxmax().date()}")
    gfc = hyg['2008-09-01':'2008-12-31']
    if len(gfc) > 0:
        print(f"  GFC worst : {gfc.min():.5f}  on {gfc.idxmin().date()}")


returns.to_csv('data/processed/returns.csv')
print(f"\nSaved: data/processed/returns.csv  {returns.shape}")

prices_df = pd.DataFrame({n: raw[n] for n in raw}).reindex(returns.index)
prices_df.to_csv('data/processed/prices.csv')
print(f"Saved: data/processed/prices.csv   {prices_df.shape}")

