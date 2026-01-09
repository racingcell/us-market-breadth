import pandas as pd
from pathlib import Path

DOCS = Path("docs")

def percent_above_ma(prices, window):
    ma = prices.rolling(window).mean()
    return (prices > ma).sum(axis=1) / prices.count(axis=1) * 100

def calc_high_low(prices, window=252):
    high = prices >= prices.rolling(window).max()
    low = prices <= prices.rolling(window).min()
    return high.sum(axis=1) - low.sum(axis=1)

def calc_ad_line(prices):
    daily_ret = prices.diff()
    adv = (daily_ret > 0).sum(axis=1)
    dec = (daily_ret < 0).sum(axis=1)
    net_adv = adv - dec
    return net_adv.cumsum()

def run_breadth(prices):
    DOCS.mkdir(exist_ok=True)

    for w in [20, 60, 120, 200]:
        pct = percent_above_ma(prices, w)
        pct.to_csv(DOCS / f"breadth_{w}.csv", header=["percent"])

    hl = calc_high_low(prices)
    hl.to_csv(DOCS / "high_low_52w.csv", header=["value"])

    ad = calc_ad_line(prices)
    ad.to_csv(DOCS / "advance_decline.csv", header=["value"])
