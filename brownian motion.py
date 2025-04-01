import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Fetch historical stock data
def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data["Close"]

# Simulate Brownian motion based on real data
def simulate_brownian_motion_with_real_data(real_prices, days, volatility):
    dt = 1  # time step
    simulated_prices = [real_prices[-1]]  # Start from the last real price
    for _ in range(days):
        change = np.random.normal(0, volatility * np.sqrt(dt))
        simulated_prices.append(simulated_prices[-1] + change)
    return simulated_prices

# Parameters
ticker = "AAPL"  # Replace with your desired stock ticker
period = "1y"  # Fetch 1 year of historical data
days = 252  # Simulate for 1 year (trading days)
volatility = 2  # Adjust based on market behavior

# Fetch real stock data
real_prices = fetch_stock_data(ticker, period)

# Validate data
if real_prices.empty:
    print(f"No data found for {ticker}. Please check the ticker symbol.")
else:
    # Simulation
    simulated_prices = simulate_brownian_motion_with_real_data(real_prices, days, volatility)
    
    # Generate future dates based on the last date in real data
    future_dates = pd.date_range(start=real_prices.index[-1], periods=days + 1, freq="B")  # "B" for business days

    # Animation Setup
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{ticker} Stock Price Simulation using Brownian Motion")
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
        return sim_line,

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(simulated_prices), interval=50, repeat=False)

    plt.show()