import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# Define the ticker symbol
ticker = "AAPL"  # Example: Apple Inc.

# Fetch stock data
stock = yf.Ticker(ticker)
stock_data = stock.history(period="100d")  # Fetch 10 years of historical stock data
earnings_data = stock.financials.loc["Net Income"]  # Fetch net income (earnings)

# Validate stock and earnings data
if stock_data.empty:
    print("Stock data is empty. Exiting...")
    exit()

if earnings_data.empty:
    print("Earnings data is empty. Exiting...")
    exit()

# Prepare stock price data
closing_prices = stock_data["Close"].resample("YE").mean()  # Use 'YE' for yearly resampling
days = closing_prices.index.day  # Extract years from the stock data

# Prepare earnings data
earnings = earnings_data.values[-len(closing_prices):]  # Slice earnings to match stock data
earnings = pd.to_numeric(earnings, errors="coerce")  # Coerce invalid data to NaN
if np.isnan(earnings).any():
    print("Earnings data contains NaN values. Cleaning data...")
    earnings = np.nan_to_num(earnings, nan=0.0)  # Replace NaN with 0

# Ensure all arrays have the same length
min_length = min(len(days), len(earnings), len(closing_prices))
days = days[:min_length]
earnings = earnings[:min_length]
closing_prices = closing_prices[:min_length]

# Debugging outputs
print(f"Days: {days}")
print(f"Earnings: {earnings}")
print(f"Closing Prices: {closing_prices}")
print(f"Earnings Min: {earnings.min()}, Earnings Max: {earnings.max()}")

# 3D Plotting setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Set axes labels
ax.set_xlabel("Days")
ax.set_ylabel("Earnings (Net Income)")
ax.set_zlabel("Stock Prices (Yearly Average)")
ax.set_title(f"3D Animation: {ticker} Earnings vs Stock Prices Over Years")

# Scatter plot initialization
scatter = ax.scatter([], [], [], c=[], cmap="viridis", s=100)

# Setting axes limits
ax.set_xlim(min(days), max(days) + 1)
ax.set_ylim(earnings.min(), earnings.max() + 10)
ax.set_zlim(closing_prices.min(), closing_prices.max() + 10)

# Animation initialization function
def init():
    scatter._offsets3d = ([], [], [])
    return scatter,

# Animation update function
def update(frame):
    x = days[:frame]
    y = earnings[:frame]
    z = closing_prices[:frame]
    scatter._offsets3d = (x, y, z)
    scatter.set_array(y)  # Color by earnings
    return scatter,

# Create animation
ani = FuncAnimation(fig, update, frames=len(days), init_func=init, interval=500)
plt.show()