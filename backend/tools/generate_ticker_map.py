# scripts/generate_ticker_map.py

import os
import yfinance as yf

def main():
    BASE = "NASDAQ100"
    tickers = [
        fn[:-4].upper()
        for fn in os.listdir(BASE)
        if fn.lower().endswith(".csv")
    ]

    alias = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            name = info.get("shortName") or info.get("longName") or t
        except Exception:
            name = t
        key = name.lower()
        for suf in (" inc.", " corp.", " corporation", " ltd."):
            key = key.replace(suf, "")
        key = key.strip()
        alias[key] = t
        alias[t.lower()] = t

    print("ticker_map = {")
    for k in sorted(alias):
        print(f'    "{k}": "{alias[k]}",')
    print("}")

if __name__ == "__main__":
    main()
