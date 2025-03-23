import csv
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Replace 'path_to_your_file.csv' with the actual path to your CSV file
file_path = r'C:\Users\LENOVO\Downloads\stockdata.csv'

data = file_path(start='2024-01-01', end='2025-01-01')

# Step 2: Calculate daily returns
data['Daily Returns'] = data['Close'].pct_change()

# Step 3: Remove NaN values
returns = data['Daily Returns'].dropna()

# Step 4: Fit a normal distribution to the returns
mu, std = norm.fit(returns)

# Step 5: Plot the PDF
plt.hist(returns, bins=50, density=True, alpha=0.6, color='g', label='Histogram of Returns')

# Overlay the fitted PDF
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(x, mu, std)
plt.plot(x, pdf, 'k', linewidth=2, label='Fitted PDF')

# Add labels and legend
plt.title(f"PDF of Daily Returns for {ticker}")
plt.xlabel("Daily Returns")
plt.ylabel("Density")
plt.legend()

# Show the plot
plt.show()
