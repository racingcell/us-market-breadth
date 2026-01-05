import FinanceDataReader as fdr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import plotly.graph_objects as go


# =========================
# CONFIG
# =========================
MARKET = "KOSPI"   # "KOSPI" or "KOSDAQ"
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
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=count_above.index,
        y=count_above.values,
        mode="lines",
        name="Stocks above 50D MA"
    )
)

fig.update_layout(
    title=f"{MARKET} – Number of Stocks Above 50-Day MA",
    xaxis_title="Date",
    yaxis_title="Count",
    template="plotly_white"
)

fig.write_html("docs/breadth_count.html", include_plotlyjs="cdn")
