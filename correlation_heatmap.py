import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Fetch data for multiple stocks
def fetch_multiple_stocks(tickers, period="1y"):
    stock_data = {}
    for ticker in tickers:
        data = yf.Ticker(ticker).history(period=period)
        stock_data[ticker] = data
    return stock_data

# Correlation Analysis
def correlation_analysis(data, tickers):
    plt.figure(figsize=(10, 8))
    returns = pd.DataFrame()
    for ticker in tickers:
        returns[ticker] = data[ticker]["Close"].pct_change()
    corr_matrix = returns.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Stock Correlation Heatmap")
    plt.show()

# Simple Price Prediction Model
def predict_stock_price(data):
    X = np.arange(len(data)).reshape(-1, 1)  # Use date indices as predictors
    y = data["Close"].values  # Stock prices as target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    plt.plot(y_test, label="Actual Prices", color="blue")
    plt.plot(predictions, label="Predicted Prices", color="red")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.show()

# Dashboard Simulation
def dashboard_simulation():
    print("Imagine a beautiful Streamlit or Dash dashboard here!")

# Main Function
if __name__ == "__main__":
    tickers = input("Enter stock tickers (comma-separated): ").split(",")
    tickers = [ticker.strip().upper() for ticker in tickers]

    data = fetch_multiple_stocks(tickers)

    # Analyze correlations
    correlation_analysis(data, tickers)

    # Predict prices for a single stock
    single_ticker = tickers[0]
    predict_stock_price(data[single_ticker])