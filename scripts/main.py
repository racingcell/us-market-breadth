from data_fetch import get_prices
from breadth_calc import run_breadth
from charts import build_charts
from ai_summary import build_summary

prices = get_prices()
run_breadth(prices)
build_charts()
build_summary()
