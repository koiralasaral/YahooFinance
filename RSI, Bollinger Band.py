import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to fetch stock data
def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Function to calculate technical indicators
def calculate_indicators(data):
    # Moving Averages
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    # RSI Calculation
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data["Middle Band"] = data["Close"].rolling(window=20).mean()
    data["Upper Band"] = data["Middle Band"] + 2 * data["Close"].rolling(window=20).std()
    data["Lower Band"] = data["Middle Band"] - 2 * data["Close"].rolling(window=20).std()

    # Daily Returns and Volatility
    data["Daily Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Daily Return"].rolling(window=30).std()

    # Sharpe Ratio (assume risk-free rate = 0 for simplicity)
    data["Sharpe Ratio"] = (data["Daily Return"].mean() / data["Volatility"])

    return data

# Function to visualize multiple stocks
def visualize_multiple_stocks(tickers, period="1y"):
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        data = fetch_stock_data(ticker, period)
        data["Cumulative Return"] = (1 + data["Close"].pct_change()).cumprod()
        plt.plot(data["Cumulative Return"], label=ticker)
    plt.title("Cumulative Returns of Selected Stocks")
    plt.legend()
    plt.show()

# Function to export data
def export_to_csv(data, ticker):
    filename = f"{ticker}_stock_analysis.csv"
    data.to_csv(filename)
    print(f"Data exported to {filename}")

# Main Function
if __name__ == "__main__":
    # Input stock tickers
    tickers = input("Enter stock tickers (comma-separated, e.g., AAPL, MSFT, TSLA): ").split(",")

    # Fetch, analyze, and visualize each stock
    for ticker in tickers:
        ticker = ticker.strip().upper()
        print(f"Analyzing {ticker}...")

        # Fetch data
        data = fetch_stock_data(ticker)

        # Validate data
        if data.empty:
            print(f"No data found for {ticker}. Skipping...")
            continue

        # Calculate indicators
        data = calculate_indicators(data)

        # Export data
        export_to_csv(data, ticker)

        # Visualize indicators
        plt.figure(figsize=(14, 7))
        plt.plot(data["Close"], label="Close Price", color="blue")
        plt.plot(data["SMA_50"], label="SMA 50", color="green", linestyle="--")
        plt.plot(data["SMA_200"], label="SMA 200", color="red", linestyle="--")
        plt.plot(data["Upper Band"], label="Upper Bollinger Band", color="orange", linestyle="--")
        plt.plot(data["Lower Band"], label="Lower Bollinger Band", color="orange", linestyle="--")
        plt.title(f"{ticker}: Stock Price and Technical Indicators")
        plt.legend()
        plt.show()

    # Compare multiple stocks
    visualize_multiple_stocks(tickers)