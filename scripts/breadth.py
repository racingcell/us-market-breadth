import os
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go
from tqdm import tqdm
from datetime import datetime

# =========================
# CONFIG
# =========================
MARKET = "KOSPI"
START_DATE = "2023-01-01"   # calculations need history
DISPLAY_START = "2024-01-01"
OUTPUT_DIR = "docs"
DATA_DIR = "docs/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def save_fig(fig, filename):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(range=[DISPLAY_START, None]),
    )
    fig.write_html(
        f"{OUTPUT_DIR}/{filename}",
        include_plotlyjs="cdn",
        config={"responsive": True}
    )

# =========================
# LOAD TICKERS
# =========================
tickers = fdr.StockListing(MARKET)["Code"].tolist()

prices = {}
for t in tqdm(tickers, desc="Downloading prices"):
    try:
        df = fdr.DataReader(t, START_DATE)
        prices[t] = df["Close"]
    except:
        pass

prices = pd.DataFrame(prices)
prices.index = pd.to_datetime(prices.index)

# =========================
# BREADTH (SMA)
# =========================
sma_periods = [20, 60, 120, 200]
breadth = {}

for p in sma_periods:
    sma = prices.rolling(p).mean()
    pct = (prices > sma).sum(axis=1) / prices.count(axis=1) * 100
    breadth[f"above_{p}"] = pct

breadth_df = pd.DataFrame(breadth)
breadth_df = breadth_df[breadth_df.index >= DISPLAY_START]
breadth_df.to_csv(f"{DATA_DIR}/breadth_sma.csv")

breadth_21dma = breadth_df.rolling(21).mean()
breadth_21dma.to_csv(f"{DATA_DIR}/breadth_sma_21dma.csv")

for col in breadth_df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=breadth_df.index, y=breadth_df[col],
                             name=col.replace("_", " ").upper()))
    fig.add_trace(go.Scatter(x=breadth_21dma.index, y=breadth_21dma[col],
                             name="21D SMA", line=dict(dash="dash")))
    fig.update_layout(title=f"KOSPI % Stocks Above {col.split('_')[1]}-Day SMA")
    save_fig(fig, f"breadth_{col.split('_')[1]}.html")

# =========================
# 52-WEEK HIGHS - LOWS
# =========================
high_52w = prices == prices.rolling(252).max()
low_52w = prices == prices.rolling(252).min()

hl_df = pd.DataFrame({
    "new_highs": high_52w.sum(axis=1),
    "new_lows": low_52w.sum(axis=1)
})
hl_df["net"] = hl_df["new_highs"] - hl_df["new_lows"]
hl_df = hl_df[hl_df.index >= DISPLAY_START]
hl_df.to_csv(f"{DATA_DIR}/high_low_52w.csv")

fig = go.Figure()
fig.add_bar(x=hl_df.index, y=hl_df["net"], name="52W Highs - Lows")
fig.update_layout(title="KOSPI 52-Week Highs minus Lows")
save_fig(fig, "high_low_52w.html")

# =========================
# ADVANCE DECLINE LINE
# =========================
returns = prices.diff()
adv = (returns > 0).sum(axis=1)
dec = (returns < 0).sum(axis=1)
net_adv = adv - dec
ad_line = net_adv.cumsum()

ad_df = pd.DataFrame({
    "advances": adv,
    "declines": dec,
    "net_advances": net_adv,
    "ad_line": ad_line
})
ad_df = ad_df[ad_df.index >= DISPLAY_START]
ad_df.to_csv(f"{DATA_DIR}/advance_decline.csv")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ad_df.index, y=ad_df["ad_line"], name="AD Line"))
fig.update_layout(title="KOSPI Advance–Decline Line")
save_fig(fig, "advance_decline.html")



# =========================
# AI SUMMARY (BREADTH + NEWS)
# =========================

import os
from openai import OpenAI
import pandas as pd

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load latest CSV data ---

# Breadth (SMA)
breadth_df = pd.read_csv(
    "docs/data/breadth_sma.csv",
    index_col=0,
    parse_dates=True
)

latest = breadth_df.iloc[-1]

breadth_20 = latest["above_20"]
breadth_60 = latest["above_60"]
breadth_120 = latest["above_120"]
breadth_200 = latest["above_200"]

# 52-week highs / lows
high_low = pd.read_csv(
    "docs/data/high_low_52w.csv",
    index_col=0,
    parse_dates=True
).iloc[-1]

# Advance–decline
ad_line = pd.read_csv(
    "docs/data/advance_decline.csv",
    index_col=0,
    parse_dates=True
).iloc[-1]


breadth_summary = f"""
Percent above moving averages:
20D: {breadth_20:.2f}%
60D: {breadth_60:.2f}%
120D: {breadth_120:.2f}%
200D: {breadth_200:.2f}%

52-week highs minus lows: {high_low['net']}
Advance–Decline line (latest): {ad_line['ad_line']}
"""


# --- Web search helper ---
def get_market_news(query):
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search"}],
        input=f"""
Summarize the most important stock market news from the last 24 hours.
Focus only on macro, earnings, policy, rates, or major risk events.
Be factual. No opinions.

Query: {query}
"""
    )
    return response.output_text.strip()

# --- Fetch news ---
us_news = get_market_news("US stock market news last 24 hours")
kr_news = get_market_news("Korean stock market news last 24 hours")

# --- Final combined AI summary ---
final_prompt = f"""
You are a professional market research assistant.

You are given:
1) Quantitative market breadth indicators
2) Current market news

Your task:
- Explain how the news context supports, contradicts, or explains the market internals
- Do NOT predict prices
- Do NOT give trading advice
- Be concise, factual, and neutral

MARKET BREADTH DATA:
{breadth_summary}

US MARKET NEWS:
{us_news}

KOREA MARKET NEWS:
{kr_news}
"""

final_response = client.responses.create(
    model="gpt-4.1-mini",
    input=final_prompt
)

summary_text = final_response.output_text.strip()

# --- Convert to HTML-safe format ---
summary_html = summary_text.replace("\n", "<br>")

# --- Write HTML file ---
with open("docs/ai_summary.html", "w", encoding="utf-8") as f:
    f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: system-ui; line-height:1.6; padding:14px;">
<h2>📊 Daily AI Market Breadth Summary</h2>
{summary_html}
</body>
</html>""")

print("AI summary with market news generated successfully.")
