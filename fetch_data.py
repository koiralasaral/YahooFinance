# Import yfinance package
import yfinance as yf
import pandas as pd
# Set the start and end date
start_date = '2015-01-01'
end_date = '2025-03-09'
# Set the ticker
tickers_list  = ['AMZN', 'AAPL', 'MSFT', 'GOOGL', 'VOW3.DE']
# Get the data 
data = pd.DataFrame(columns=tickers_list)
# Fetch the data
for ticker in tickers_list:
    data[ticker] = yf.download(ticker, start_date, end_date)['Close']
# Print 5 rows
print(data.head())
# Print 5 rows
print(data.tail())