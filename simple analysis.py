import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Fetch Stock Data
def fetch_stock_data(ticker, period="1y"):
    """
    Fetches historical stock data for a given ticker.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Step 2: Calculate Technical Indicators
def calculate_indicators(data):
    """
    Adds technical indicators to the stock data.
    """
    # Moving Averages
    data["SMA_50"] = data["Close"].rolling(window=50).mean()  # Simple Moving Average (50 days)
    data["SMA_200"] = data["Close"].rolling(window=200).mean()  # Simple Moving Average (200 days)

    # Daily Returns
    data["Daily Return"] = data["Close"].pct_change()

    # Cumulative Return
    data["Cumulative Return"] = (1 + data["Daily Return"]).cumprod()

    return data

# Step 3: Visualize the Data
def visualize_data(data, ticker):
    """
    Visualizes stock data and indicators.
    """
    plt.figure(figsize=(14, 7))

    # Plot Closing Prices and Moving Averages
    plt.subplot(2, 1, 1)
    plt.plot(data["Close"], label="Close Price", color="blue")
    plt.plot(data["SMA_50"], label="SMA 50", color="green", linestyle="--")
    plt.plot(data["SMA_200"], label="SMA 200", color="red", linestyle="--")
    plt.title(f"{ticker}: Stock Price and Moving Averages")
    plt.legend()

    # Plot Cumulative Returns
    plt.subplot(2, 1, 2)
    plt.plot(data["Cumulative Return"], label="Cumulative Return", color="purple")
    plt.title(f"{ticker}: Cumulative Return")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Step 4: Analyze the Stock
def analyze_stock(ticker):
    """
    Fetches, calculates indicators, and visualizes data for a given stock ticker.
    """
    print(f"Analyzing {ticker}...")

    # Fetch data
    data = fetch_stock_data(ticker)

    # Validate data
    if data.empty:
        print(f"No data found for {ticker}. Please check the ticker symbol.")
        return

    # Add indicators
    data = calculate_indicators(data)

    # Visualize data
    visualize_data(data, ticker)

# Main Function
if __name__ == "__main__":
    # Enter the stock ticker (e.g., AAPL, MSFT, TSLA, etc.)
    ticker_symbol = input("Enter the stock ticker symbol: ").upper()
    analyze_stock(ticker_symbol)