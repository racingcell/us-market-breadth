import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# =========================
# CONFIG
# =========================
MARKET = "KOSDAQ"   # "KOSPI" or "KOSDAQ"
START_DATE = "2024-01-01"
MA_WINDOW = 50
OUTPUT_DIR = Path("docs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# FETCH LIST
# =========================
stocks = fdr.StockListing(MARKET)
tickers = stocks["Code"].tolist()

price_data = {}

for ticker in tqdm(tickers):
    try:
        df = fdr.DataReader(ticker, START_DATE).tail(200)
        if len(df) >= MA_WINDOW:
            price_data[ticker] = df["Close"]
    except Exception:
        continue

prices = pd.DataFrame(price_data).sort_index()
ma50 = prices.rolling(MA_WINDOW).mean()

count_above = (prices > ma50).sum(axis=1)
percent_above = count_above / prices.count(axis=1) * 100

# =========================
# SAVE PLOTS
# =========================
plt.figure(figsize=(12, 6))
plt.plot(count_above)
plt.title(f"{MARKET} – Number of Stocks Above 50D MA")
plt.xlabel("Date")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "breadth_count.png")
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(percent_above)
plt.title(f"{MARKET} – % of Stocks Above 50D MA")
plt.xlabel("Date")
plt.ylabel("Percent")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "breadth_percent.png")
plt.close()

print("Plots saved to docs/")
