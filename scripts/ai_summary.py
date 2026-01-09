import pandas as pd
from pathlib import Path
from openai import OpenAI
import os

DOCS = Path("docs")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_summary():
    # --- Load S&P 500 market breadth data ---
    data = {}
    for w in [20, 60, 120, 200]:
        df = pd.read_csv(DOCS / f"breadth_{w}.csv")
        data[f"{w}d"] = round(df.iloc[-1, 1], 2)

    hl = pd.read_csv(DOCS / "high_low_52w.csv").iloc[-1, 1]
    ad = pd.read_csv(DOCS / "advance_decline.csv").iloc[-1, 1]

    breadth_summary = f"""
S&P 500 Market Breadth

Percent of stocks above moving averages:
• 20-day: {data['20d']}%
• 60-day: {data['60d']}%
• 120-day: {data['120d']}%
• 200-day: {data['200d']}%

52-week highs minus lows: {hl}
Advance-Decline line level: {ad}
"""

    # --- Fetch recent US market news using search-preview model ---
    # NOTE: search-preview models do NOT support temperature, top_p, etc.
    news_response = client.chat.completions.create(
        model="gpt-4o-mini-search-preview",
        messages=[{
            "role": "user",
            "content": """
Summarize the most important US stock market news from the last 24 hours.
Focus only on macro events, major corporate earnings, policy announcements, interest rates, or significant risk events.
Be concise, factual, and neutral.
If no major news, say: "No significant US market-moving news in the last 24 hours."
"""
        }]
    )

    us_news = news_response.choices[0].message.content.strip()

    # --- Final combined AI summary ---
    final_prompt = f"""
You are a professional, neutral market analyst.

Current S&P 500 market breadth internals:
{breadth_summary.strip()}

Recent US market news context (last 24 hours):
{us_news}

Write a concise professional market summary in plain text (no markdown):
- Briefly describe the current breadth readings
- Mention how the recent news may support, contradict, or provide background for these internals (if relevant)
- Remain strictly factual and neutral
- Do not predict future price movements
- Do not give trading advice

Provide the market summary first in English and then in Korean
"""

    final_response = client.chat.completions.create(
        model="gpt-4o",  # Supports temperature and gives excellent writing quality
        temperature=0.3,
        messages=[{"role": "user", "content": final_prompt}]
    )

    summary_text = final_response.choices[0].message.content.strip()

    # --- Convert to HTML-safe format ---
    summary_html = summary_text.replace("\n", "<br>")

    # --- Write HTML file ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: system-ui, sans-serif; line-height: 1.6; padding: 16px; max-width: 800px; margin: 0 auto; }}
        h2 {{ margin-top: 32px; }}
    </style>
</head>
<body>
    <h2>S&P 500 Daily Breadth & News Summary</h2>
    {summary_html}
</body>
</html>"""

    (DOCS / "ai_summary.html").write_text(html, encoding="utf-8")
    print("AI summary with real-time US news generated: docs/ai_summary.html")


if __name__ == "__main__":
    build_summary()