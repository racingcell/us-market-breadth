import FinanceDataReader as fdr
import pandas as pd

MARKET = "S&P500"

def get_prices(start="2024-01-01"):
    listings = fdr.StockListing(MARKET)

    # Robust symbol column handling
    symbol_col = "Symbol" if "Symbol" in listings.columns else listings.columns[0]
    tickers = listings[symbol_col].dropna().unique().tolist()

    prices = []
    for t in tickers:
        try:
            df = fdr.DataReader(t, start=start)[["Close"]]
            df.rename(columns={"Close": t}, inplace=True)
            prices.append(df)
        except Exception:
            continue

    prices = pd.concat(prices, axis=1)
    prices = prices.dropna(how="all")
    return prices