import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import math
# Download stock market data from Yahoo Finance
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2021-01-01')
data.reset_index(inplace=True)
data.to_csv('stock_data.csv', index=False)
# Load stock market data from a CSV file
data = pd.read_csv('stock_data.csv')

# Display the first few rows of the data
print(data.head())
# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])
# Calculate the moving average
# Calculate the average closing price
average_close = data['Close'].mean()
print(f"The average closing price is: {average_close}")
# Plot the closing prices and moving average
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close Price')  
plt.axhline(y=average_close, color='r', linestyle='-', label='Moving Average')
plt.title(f'{ticker} Close Price and Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
# Calculate the daily returns
data['Daily Return'] = data['Close'].pct_change()
# Calculate the mean of the daily returns
mean_daily_returns = data['Daily Return'].mean()
print(f"The mean of the daily returns is: {mean_daily_returns}")
# Calculate the standard deviation of the daily returns
std_daily_returns = data['Daily Return'].std()
print(f"The standard deviation of the daily returns is: {std_daily_returns}")
# Plot the daily returns
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Daily Return'], label='Daily Returns')
plt.title(f'{ticker} Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.legend()
plt.show()
# Calculate the annualized return