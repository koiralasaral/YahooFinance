import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Fetch historical stock data
def fetch_stock_data(ticker, period="10y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data["Close"]

# Calculate current volatility
def calculate_volatility(real_prices):
    daily_returns = real_prices.pct_change()  # Calculate daily percentage changes
    annual_volatility = daily_returns.std() * np.sqrt(252)  # Annualize volatility (252 trading days)
    return annual_volatility

# Simulate Brownian motion based on real data
def simulate_brownian_motion_with_real_data(real_prices, days, volatility):
    dt = 1  # time step
    simulated_prices = [real_prices[-1]]  # Start from the last real price
    for _ in range(days):
        change = np.random.normal(0, volatility * np.sqrt(dt))
        simulated_prices.append(simulated_prices[-1] + change)
    return simulated_prices

# Parameters
ticker = "NGT.TO"  # Replace with your desired stock ticker
period = "10y"  # Fetch 1 year of historical data
days = 1000  # Simulate for the next 3 months (approximately 63 trading days)

# Fetch real stock data
real_prices = fetch_stock_data(ticker, period)

# Validate data
if real_prices.empty:
    print(f"No data found for {ticker}. Please check the ticker symbol.")
else:
    # Calculate current volatility
    current_volatility = calculate_volatility(real_prices)
    print(f"Current Volatility for {ticker}: {current_volatility:.2f}")

    # Simulation
    simulated_prices = simulate_brownian_motion_with_real_data(real_prices, days, current_volatility)
    
    # Generate future dates based on the last date in real data
    future_dates = pd.date_range(start=real_prices.index[-1], periods=days + 1, freq="B")  # "B" for business days

    # Animation Setup
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{ticker} Stock Price Simulation for the Next 3 Months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    
    # Line plot for real prices
    ax.plot(real_prices.index, real_prices, label="Real Prices", color="blue")

    # Initialize line for simulated prices
    sim_line, = ax.plot([], [], label="Simulated Prices", color="orange")
    ax.legend()

    # Animation function
    def update(frame):
        sim_line.set_data(future_dates[:frame + 1], simulated_prices[:frame + 1])
        # Dynamically adjust X and Y limits
        ax.set_xlim(real_prices.index[0], future_dates[-1])  # Extend X-axis from historical to future dates
        ax.set_ylim(min(real_prices.min(), min(simulated_prices)), max(real_prices.max(), max(simulated_prices)))  # Adjust Y-axis
        return sim_line,

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(simulated_prices), interval=50, repeat=False)

    plt.show()