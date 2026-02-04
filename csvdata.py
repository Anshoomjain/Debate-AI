import yfinance as yf
data = yf.download("HG=F" \
"", start="2016-01-01", end="2026-01-01")
data.to_csv("copper.csv")   