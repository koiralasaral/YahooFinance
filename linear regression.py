import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fetch Apple stock data
ticker = "BHARTIARTL.NS"  # Apple Inc.
data = yf.download(ticker, period="900d")

# Validate data
if data.empty:
    print(f"No data found for ticker: {ticker}")
    exit()



# Prepare data for regression
closing_prices = data['Close'].values
days = np.arange(len(closing_prices)).reshape(-1, 1)  # Days as independent variable
prices = closing_prices.reshape(-1, 1)  # Closing prices as dependent variable

# Perform Linear Regression
model = LinearRegression()
model.fit(days, prices)

# Coefficients of the regression line
slope = model.coef_[0][0]
intercept = model.intercept_[0]

# Display the regression equation
print(f"Regression Equation: y = {slope:.2f}x + {intercept:.2f}")

# Generate predictions
predicted_prices = model.predict(days)

# Plot the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(days, prices, color='blue', label='Actual Prices')
plt.plot(days, predicted_prices, color='red', linewidth=2, label='Regression Line')
plt.title(f"Linear Regression for {ticker}")
plt.xlabel("Days")
plt.ylabel("Stock Closing Price")
plt.legend()
plt.grid()
plt.show()