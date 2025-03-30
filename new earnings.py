import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# Define the ticker symbols for 7 companies
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX"]

# Initialize lists to store data
company_names = []
stock_prices = []
earnings = []

# Fetch data for each ticker
for ticker in tickers:
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="1y")  # Fetch 1 year of historical stock data
    earnings_data = stock.financials.loc["Net Income"]  # Fetch net income (earnings)

    # Validate data
    if stock_data.empty or earnings_data.empty:
        print(f"Data not available for {ticker}. Skipping...")
        continue

    # Prepare stock price data
    avg_price = stock_data["Close"].mean()  # Average closing price
    net_income = earnings_data.values[0]  # Latest net income

    # Append data
    company_names.append(ticker)
    stock_prices.append(avg_price)
    earnings.append(net_income)

# Convert data to numpy arrays for plotting
stock_prices = np.array(stock_prices)
earnings = np.array(earnings)
company_indices = np.arange(len(company_names))  # Numeric indices for companies

# 3D Plotting setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot
scatter = ax.scatter(stock_prices, earnings, company_indices, c=earnings, cmap="viridis", s=100)

# Set axes labels
ax.set_xlabel("Stock Prices (Average Closing Price)")
ax.set_ylabel("Earnings (Net Income)")
ax.set_zlabel("Company Index")
ax.set_title("3D Plot: Earnings vs Stock Prices for Companies")

# Add company names as annotations
for idx, company in enumerate(company_names):
    ax.text(stock_prices[idx], earnings[idx], company_indices[idx], company, color="black")

plt.colorbar(scatter, label="Earnings (Net Income)")
plt.show()