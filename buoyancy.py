import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yfinance as yf

# Define tickers for CIBC and Osisko Gold Royalties
tickers = ["CM.TO", "OR.TO"]  # TSX tickers for CIBC and Osisko Gold Royalties

# Fetch data for both tickers
data_cibc = yf.download(tickers[0], start="2023-01-01", end="2025-03-30")
data_osisko = yf.download(tickers[1], start="2023-01-01", end="2025-03-30")

# Extract closing prices
closing_prices_cibc = data_cibc['Close'].values.flatten()
closing_prices_osisko = data_osisko['Close'].values.flatten()

# Ensure consistent lengths by trimming to the shorter dataset
min_length = min(len(closing_prices_cibc), len(closing_prices_osisko))
closing_prices_cibc = closing_prices_cibc[:min_length]
closing_prices_osisko = closing_prices_osisko[:min_length]

# Create consistent 1D arrays for x, y, and z
x = np.arange(min_length)  # Stock index positions
y_cibc = np.zeros(min_length)  # Align CIBC along one Y-axis
y_osisko = np.ones(min_length) * 5  # Offset Osisko on a different Y-axis

# Animation setup
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize scatter plot for both stocks
scatter_cibc = ax.scatter(x, y_cibc, closing_prices_cibc, s=80, c=closing_prices_cibc, cmap='Blues', label="CIBC")
scatter_osisko = ax.scatter(x, y_osisko, closing_prices_osisko, s=80, c=closing_prices_osisko, cmap='Oranges', label="Osisko Gold Royalties")

# Update function for animation
def update(frame):
    new_z_cibc = closing_prices_cibc + np.sin(frame / 10) * 5
    new_z_osisko = closing_prices_osisko + np.sin(frame / 10) * 5
    scatter_cibc._offsets3d = (x, y_cibc, new_z_cibc)
    scatter_osisko._offsets3d = (x, y_osisko, new_z_osisko)
    scatter_cibc.set_array(new_z_cibc)
    scatter_osisko.set_array(new_z_osisko)
    return scatter_cibc, scatter_osisko

# Set plot limits and labels
ax.set_xlim(0, min_length)
ax.set_ylim(-5, 10)
ax.set_zlim(min(min(closing_prices_cibc), min(closing_prices_osisko)) - 10, max(max(closing_prices_cibc), max(closing_prices_osisko)) + 10)
ax.set_xlabel("Stock Index")
ax.set_ylabel("Market Plane")
ax.set_zlabel("Buoyancy Level (Closing Price)")
ax.set_title("Stock Buoyancy Animation for CIBC and Osisko Gold Royalties")
ax.legend()

# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=50)
plt.show()