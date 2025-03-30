import yfinance as yf
import matplotlib.pyplot as plt

# Fetch stock data
ticker = "GC=F"  # Example: Apple Inc.
data = yf.download(ticker, start="2023-01-01", end="2025-03-30")

# Calculate rate of change (akin to induced EMF)
data['Rate_of_Change'] = data['Close'].diff()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Stock Price (Close)', color='blue')
plt.plot(data.index, data['Rate_of_Change'], label='Rate of Change', color='orange')
plt.title(f"{ticker} Stock Price and Rate of Change")
plt.xlabel("Date")
plt.ylabel("Price / Rate of Change")
plt.legend()
plt.grid()
plt.show()