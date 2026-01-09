import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DOCS = Path("docs")
START = "2024-01-01"

def save_breadth_chart(window):
    df = pd.read_csv(DOCS / f"breadth_{window}.csv", index_col=0, parse_dates=True)
    sma21 = df["percent"].rolling(21).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["percent"], name=f"% > {window}D MA"))
    fig.add_trace(go.Scatter(x=df.index, y=sma21, name="21D SMA"))

    fig.update_layout(
        height=900,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(range=[START, df.index.max()]),
        autosize=True
    )

    fig.write_html(DOCS / f"breadth_{window}.html", include_plotlyjs="cdn")

def save_simple_chart(csv, out, height=700):
    df = pd.read_csv(DOCS / csv, index_col=0, parse_dates=True)

    fig = go.Figure(go.Bar(x=df.index, y=df.iloc[:, 0]))
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(range=[START, df.index.max()]),
        autosize=True
    )

    fig.write_html(DOCS / out, include_plotlyjs="cdn")

def build_charts():
    for w in [20, 60, 120, 200]:
        save_breadth_chart(w)

    save_simple_chart("high_low_52w.csv", "high_low_52w.html")
    save_simple_chart("advance_decline.csv", "advance_decline.html")
