import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression

# Fetch stock data
ticker = "GC=F"  # Example: Apple Inc.
data = yf.download(ticker, start="2019-01-01", end="2025-03-30")

# Validate data
if data.empty:
    print(f"No data found for ticker: {ticker}")
    exit()

# Prepare data for regression analysis
closing_prices = data['Close'].values
days = np.arange(len(closing_prices)).reshape(-1, 1)  # Days as independent variable
prices = closing_prices.reshape(-1, 1)  # Closing prices as dependent variable

# Initialize regression model
model = LinearRegression()

# Animation setup
fig, ax = plt.subplots(figsize=(10, 6))

# Adjusting the axes based on the data
ax.set_xlim(0, len(days))
ax.set_ylim(min(prices) - 10, max(prices) + 10)

line, = ax.plot([], [], color="red", lw=2, label="Regression Line")
scatter = ax.scatter(days, prices, color="blue", label="Data Points")
ax.set_title(f"Linear Regression Animation for {ticker}")
ax.set_xlabel("Days")
ax.set_ylabel("Stock Prices")
ax.legend()

# Text object to display the equation
equation_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Function to initialize the animation
def init():
    line.set_data([], [])
    equation_text.set_text("")
    return line, equation_text

# Update function for animation
def update(frame):
    # Fit the model up to the current frame (not all points)
    model.fit(days[:frame], prices[:frame])  # Fit model on first `frame` samples
    predictions = model.predict(days)  # Predict regression line for all days
    line.set_data(days, predictions)  # Update the line with new predictions
    
    # Get the current regression coefficients (slope and intercept)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    # Equation in the form y = mx + b
    equation = f"y = {slope:.2f}x + {intercept:.2f}"
    
    # Update the displayed equation
    equation_text.set_text(f"Regression Line: {equation}")
    
    return line, equation_text

# Limit the number of frames for faster animation (optional)
frame_limit = min(len(days), 500)  # Limit to first 500 data points

# Create animation
ani = FuncAnimation(fig, update, frames=frame_limit, init_func=init, blit=True, interval=50)

# Display the animation
plt.show()
