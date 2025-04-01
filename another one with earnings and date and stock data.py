import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

# Define tickers for 7 companies
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX"]

# Initialize lists to store combined data
all_dates = []
all_prices = []
all_earnings = []
all_companies = []

# Fetch data for each ticker
for ticker in tickers:
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="10d")  # Fetch last 10 days of historical stock data
    earnings_data = stock.financials.loc["Net Income"]  # Fetch net income (earnings)

    # Validate data
    if stock_data.empty or earnings_data.empty:
        print(f"Data not available for {ticker}. Skipping...")
        continue

    # Extract data
    dates = stock_data.index  # Dates of stock prices
    prices = stock_data["Close"].values  # Closing stock prices
    net_income = earnings_data.values[0]  # Latest net income (assume one value for simplicity)

    # Fill earnings data for all dates
    earnings = np.full(len(dates), net_income)

    # Append data to the combined lists
    all_dates.extend(dates)
    all_prices.extend(prices)
    all_earnings.extend(earnings)
    all_companies.extend([ticker] * len(dates))  # Repeat the ticker for each date

# Convert lists to numpy arrays for plotting
date_indices = np.arange(len(all_dates))  # Convert dates to numeric indices
all_prices = np.array(all_prices)
all_earnings = np.array(all_earnings)

# 3D Plotting setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Initialize scatter plot
scatter = ax.scatter([], [], [], c=[], cmap="viridis", s=100)

# Set axes labels
ax.set_xlabel("Date Index")
ax.set_ylabel("Earnings (Net Income)")
ax.set_zlabel("Stock Prices (Closing Price)")
ax.set_title("Animated 3D Plot: Earnings vs Stock Prices Over Dates")

# Define update function for animation
def update(frame):
    ax.clear()  # Clear the previous plot
    ax.set_xlabel("Date Index")
    ax.set_ylabel("Earnings (Net Income)")
    ax.set_zlabel("Stock Prices (Closing Price)")
    ax.set_title("Animated 3D Plot: Earnings vs Stock Prices Over Dates")
    scatter = ax.scatter(
        date_indices[:frame], 
        all_earnings[:frame], 
        all_prices[:frame], 
        c=all_prices[:frame], 
        cmap="viridis", 
        s=100
    )
    return scatter,

# Create animation
ani = FuncAnimation(fig, update, frames=len(all_dates), interval=200, repeat=True)

plt.show()