import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

###############################################
# PART 1: Download Data for 10 Companies
###############################################
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'IBM', 'ORCL', 'CSCO', 'INTC']
start_date = "2018-01-01"
end_date   = "2023-01-01"

print("Downloading monthly closing prices for companies:")
print(companies)

# Download daily closing prices and keep the 'Close' column.
# (Note: By default, yfinance auto_adjust is now True.)
data_all = yf.download(companies, start=start_date, end=end_date)['Close']

# Resample to monthly data using Month End frequency ('ME') to avoid deprecation warnings.
data_monthly = data_all.resample('ME').last().dropna()

# Print a summary of the downloaded data.
print("\nDownloaded Monthly Data (first 5 rows):")
print(data_monthly.head())

###############################################
# PART 2: Compute Sequence and Nested Intervals 
###############################################
# For demonstration, we will compute the sequence for one chosen company.
# (You could loop over all companies to compare sequences if desired.)
# We choose AAPL from the list.
chosen_company = "AAPL"
print(f"\nUsing {chosen_company} for the sequence & nested intervals demonstration.")

# Extract the monthly series for the chosen company.
series = data_monthly[chosen_company]

# Compute log returns: r_t = ln(P_t/P_{t-1})
log_returns = np.log(series / series.shift(1)).dropna()

# Compute cumulative (expanding) average of the log returns:
#   x_n = (r_1 + r_2 + ... + r_n)/n.
cum_avg = log_returns.expanding().mean()

# Convert to a one-dimensional array of Python floats.
x = cum_avg.to_numpy().flatten()
n_vals = np.arange(1, len(x) + 1)

print("\n=== Cumulative Average Log Returns Sequence for", chosen_company, "===")
for i, val in enumerate(x):
    print(f"n = {i+1:2d} : x_{i+1} = {float(val):.6f}")

# Define individual intervals and compute the intersection (nested intervals).
# For each x_i in the sequence, define an interval:
#    I_i = [ x_i - ε_i,  x_i + ε_i ]  where ε_i = 1/(i+2)
def interval_intersection(intervals):
    lower = max(iv[0] for iv in intervals)
    upper = min(iv[1] for iv in intervals)
    return (lower, upper)

nested_intervals = []  # This will hold I_1, I_2, …, I_n.
current_intervals = []

print("\n=== Nested Intervals Computation (for " + chosen_company + ") ===")
for i, xi in enumerate(x):
    epsilon = 1.0 / (i + 2)  # ε_i = 1/(i+2)
    new_interval = (xi - epsilon, xi + epsilon)
    current_intervals.append(new_interval)
    I_n = interval_intersection(current_intervals)
    nested_intervals.append(I_n)
    width = I_n[1] - I_n[0]
    print(f"n = {i+1:2d}: x = {xi:.6f}, new interval = [{new_interval[0]:.6f}, {new_interval[1]:.6f}], "
          f"I_{i+1} = [{I_n[0]:.6f}, {I_n[1]:.6f}], width = {width:.6f}")

print("\nAxiom of Completeness states: If the widths of these nested intervals tend to zero, their intersection is a single point.")
print("Final Intersection (Approximate Limit):", nested_intervals[-1])

###############################################
# PART 3: 3D Animation of the Nested Intervals
###############################################
# In the 3D animation:
# - The x-axis represents the real number line (the value).
# - For each interval I_n = [lower, upper] (for n = 1, 2, ..., N),
#   we plot two points at fixed y coordinates: one at (lower, 0, n) and one at (upper, 1, n).
#   Then we draw a segment connecting these two points.
# - The z-axis is used for the index n.
N = len(nested_intervals)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Value")
ax.set_ylabel("Endpoint (0: Lower, 1: Upper)")
ax.set_zlabel("Index n")
ax.set_title("3D Animation of Nested Intervals for " + chosen_company)

# The update function for animation will clear the axis and re-plot intervals from 1 to frame.
def update_3d(frame):
    ax.cla()  # Clear the axes for new drawing.
    ax.set_xlabel("Value")
    ax.set_ylabel("Endpoint (0: Lower, 1: Upper)")
    ax.set_zlabel("Index n")
    ax.set_title("3D Animation of Nested Intervals for " + chosen_company)
    
    # Plot intervals from index 1 to 'frame'
    for i in range(frame):
        low, up = nested_intervals[i]
        n_idx = i + 1  # index starting at 1
        # Plot the line between the two endpoints.
        ax.plot([low, up], [0, 1], [n_idx, n_idx],
                color='blue', lw=3, marker='o')
        # Annotate the width in red
        width = up - low
        ax.text((low+up)/2, 0.5, n_idx, f"{width:.6f}", fontsize=8, color='red')
    
    # Set limits based on overall min/max.
    all_lows = [iv[0] for iv in nested_intervals[:frame]]
    all_ups  = [iv[1] for iv in nested_intervals[:frame]]
    if all_lows and all_ups:
        ax.set_xlim(min(all_lows)-0.001, max(all_ups)+0.001)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(0, N + 1)
    return ax,

ani3d = FuncAnimation(fig, update_3d, frames=N, interval=500, blit=False)
plt.show()