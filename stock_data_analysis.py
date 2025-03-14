# Import yfinance package
import yfinance as yf
 # Set the start date and end data
import matplotlib.pyplot as plt
import pandas as pd                 

start_date = '2025-01-18'
end_date = '2025-03-09'
# Set the ticker symbol
# Set the ticker
tickers_list  = ['AMZN', 'AAPL', 'MSFT', 'GOOGL', 'VOW3.DE']
# Get the data 
data = pd.DataFrame(columns=tickers_list)
# Fetch the data
for ticker in tickers_list:
    data[ticker] = yf.download(ticker, start_date, end_date)['Close']
# Get the closing price
closing_price = data['Close']
# Print the closing price
print(f"Closing Price: {closing_price}")
# Plot the closing price
plt.plot(closing_price)
# Add a title
plt.title('Closing Price')
# Add an x-axis label       
plt.xlabel('Date')
# Add a y-axis label
plt.ylabel('Closing Price')
# Show the plot
plt.show()
# Calculate the daily returns
daily_returns = closing_price.pct_change()
# Print the daily returns
print(f"Daily Returns: {daily_returns}")
# Plot the daily returns    
plt.plot(daily_returns)
plt.title('Daily Returns')  # Add a title
plt.xlabel('Date')  # Add an x-axis label                   
plt.ylabel('Daily Returns')  # Add a y-axis label
plt.grid()
plt.show()  
# Show the plot                 
plt.show()
# Calculate the mean of the daily returns
mean_daily_returns = daily_returns.mean()
# Print the mean of the daily returns   


print(f"Mean of daily Returns: {mean_daily_returns}")
      
plt.plot(mean_daily_returns)
plt.title(f"Mean of daily Returns: {mean_daily_returns}")  # Add a title
plt.xlabel('Date')  # Add an x-axis label                   
plt.ylabel('Daily Returns')  # Add a y-axis label
plt.grid()
plt.show()  
# Show the plot                 
plt.show()
# Print the standard deviation of the daily returns
std_daily_returns = daily_returns.std()
print(f"Standard Deviation of the daily returns: {std_daily_returns}")
plt.plot(std_daily_returns)
plt.title(f' Standard Deviation of the daily returns: {std_daily_returns}')  # Add a title
plt.xlabel('Date')  # Add an x-axis label                   
plt.ylabel('Daily Returns')  # Add a y-axis label
plt.grid()
plt.show()  
# Show the plot                 
plt.show()
# Calculate the correlation between the daily returns
correlation = daily_returns.corr()                      
# Print the correlation
print(f"Correlation: {correlation}")                
# Plot the correlation
plt.plot(correlation)
plt.title(f'Correlation: {correlation}')  # Add a title 
plt.xlabel('Date')  # Add an x-axis label
plt.ylabel('Daily Returns')  # Add a y-axis label
plt.grid()
plt.show()
# Show the plot
plt.show()
# Calculate the covariance between the daily returns
covariance = daily_returns.cov()
# Print the covariance
print(f"Covariance: {covariance}")
# Plot the covariance
plt.plot(covariance)
plt.title(f'Covariance: {covariance}')  # Add a title
plt.xlabel('Date')  # Add an x-axis label
plt.ylabel('Daily Returns')  # Add a y-axis label
plt.grid()
plt.show()
# Show the plot
plt.show()
