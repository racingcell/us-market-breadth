import pandas as pd
from pathlib import Path
from openai import OpenAI
import os

DOCS = Path("docs")

def build_summary():
    data = {}
    for w in [20, 60, 120, 200]:
        df = pd.read_csv(DOCS / f"breadth_{w}.csv")
        data[f"{w}d"] = round(df.iloc[-1, 1], 2)

    hl = pd.read_csv(DOCS / "high_low_52w.csv").iloc[-1, 1]
    ad = pd.read_csv(DOCS / "advance_decline.csv").iloc[-1, 1]

    prompt = f"""
S&P500 market breadth summary.

Percent above MAs:
20d {data['20d']}%
60d {data['60d']}%
120d {data['120d']}%
200d {data['200d']}%

52w highs minus lows: {hl}
Advance Decline line level: {ad}

Write a concise professional market summary. No markdown. Plain text.
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    text = resp.choices[0].message.content.strip()

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: system-ui; padding: 12px; }}
</style>
</head>
<body>
{text.replace('\n', '<br>')}
</body>
</html>
"""

    (DOCS / "ai_summary.html").write_text(html, encoding="utf-8")
