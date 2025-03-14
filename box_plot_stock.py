import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Set the start date and end date
start_date = '2025-01-18'
end_date = '2025-03-09'

# Set the ticker symbols
tickers_list = ['AMZN', 'AAPL', 'MSFT', 'GOOGL', 'VOW3.DE']

# Fetch the data and store the closing prices in a DataFrame
data = pd.DataFrame()
for ticker in tickers_list:
    data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Close']

# Plot the closing prices for all tickers
data.plot(figsize=(10,6))
plt.title('Closing Prices of Stock Tickers')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend(tickers_list)
plt.grid()
plt.show()

# Calculate the daily returns for each ticker
daily_returns = data.pct_change()

# Plot the daily returns for all tickers
daily_returns.plot(figsize=(10,6))
plt.title('Daily Returns of Stock Tickers')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.legend(tickers_list)
plt.grid()
plt.show()

# Calculate and print the mean of the daily returns for each ticker
mean_daily_returns = daily_returns.mean()
print(f"Mean of daily Returns: \n{mean_daily_returns}")

# Plot the mean of the daily returns
mean_daily_returns.plot(kind='bar', figsize=(10,6))
plt.title('Mean Daily Returns of Stock Tickers')
plt.xlabel('Ticker')
plt.ylabel('Mean Daily Return')
plt.grid()
plt.show()

# Calculate and print the standard deviation of the daily returns for each ticker
std_daily_returns = daily_returns.std()
print(f"Standard Deviation of Daily Returns: \n{std_daily_returns}")

# Plot the standard deviation of daily returns
std_daily_returns.plot(kind='bar', figsize=(10,6))
plt.title('Standard Deviation of Daily Returns of Stock Tickers')
plt.xlabel('Ticker')
plt.ylabel('Standard Deviation of Daily Return')
plt.grid()
plt.show()

# Calculate and print the correlation matrix between daily returns
correlation = daily_returns.corr()
print(f"Correlation Matrix of Daily Returns: \n{correlation}")

# Plot the correlation matrix
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Daily Returns')
plt.show()

# Calculate and print the covariance matrix between daily returns
covariance = daily_returns.cov()
print(f"Covariance Matrix of Daily Returns: \n{covariance}")

# Plot the covariance matrix
plt.figure(figsize=(8,6))
sns.heatmap(covariance, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Covariance Matrix of Daily Returns')
plt.show()