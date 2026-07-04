

import pandas as pd
import yfinance as yf
import os
import time


for d in ['data/raw', 'data/processed',
          'results/predictions', 'results/tables', 'results/figures']:
    os.makedirs(d, exist_ok=True)


START = '2005-01-01'
END   = '2024-12-31'

print(f"Downloading {START} to {END}  (includes 2008 GFC)\n")

def download(ticker, name, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=START, end=END, progress=False)
            if df.empty:
                print(f"  WARNING {name} ({ticker}): empty dataframe")
                return None
            df.to_csv(f'data/raw/{name}.csv')
            print(f"  {name:20s}  {len(df):5d} rows  "
                  f"{df.index.min().date()} -> {df.index.max().date()}")
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  FAILED {name} ({ticker}): {e}")
                return None


print("=== CORE ASSETS ===")
download('^GSPC',    'sp500')
download('^TNX',     'treasury')
download('EURUSD=X', 'eurusd')
download('CL=F',     'crude_oil')
download('^VIX',     'vix')
download('HYG',      'credit_spread')   # HYG = credit spread proxy


print("\n=== CROSS-ASSET FEATURES ===")
dxy = download('DX-Y.NYB', 'dollar_index')   # correct ICE DXY ticker
if dxy is None:
    print("  dollar_index failed — data_cleaning.py will build -EURUSD proxy")
    pd.DataFrame().to_csv('data/raw/dollar_index.csv')

download('GC=F',     'gold')
download('HG=F',     'copper')
download('JPYUSD=X', 'jpyusd')


print("\n=== QUALITY SUMMARY ===")
files = ['sp500','treasury','eurusd','crude_oil','vix',
         'credit_spread','dollar_index','gold','copper','jpyusd']
all_ok = True
for name in files:
    path = f'data/raw/{name}.csv'
    try:
        n = len(pd.read_csv(path))
        status = "OK" if n > 100 else "WARNING: very few rows"
        print(f"  {name:20s}  {n:5d} rows  {status}")
        if n <= 100:
            all_ok = False
    except FileNotFoundError:
        print(f"  {name:20s}  FILE NOT FOUND")
        all_ok = False

print(f"\n{'All files ready' if all_ok else 'Some files missing — check warnings above'}")
print("Next step: python data_cleaning.py")