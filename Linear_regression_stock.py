import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Example: downloading data for one ticker; adjust as needed.
ticker = "AAPL"
data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)

if data.empty:
    print(f"Data for {ticker} is empty. Check the ticker or try again later.")
else:
    # Assuming you use the "Close" price
    data = data[['Close']]
    data['Daily_Return'] = data['Close'].pct_change().fillna(0)
    
    # Prepare data for regression: here we use the day index as X and Close price as y.
    n_points = len(data)
    x = np.arange(n_points).reshape(-1, 1)
    y = data['Close'].values
    
    # Debug prints: check the shape of x and y.
    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)
    
    if x.shape[0] == 0 or y.shape[0] == 0:
        print("No data points available for regression. Exiting the fitting process.")
    else:
        model = LinearRegression()
        model.fit(x, y)
        print("Regression Model Parameters:")
        print("Intercept:", model.intercept_)
        print("Slope:", model.coef_[0])