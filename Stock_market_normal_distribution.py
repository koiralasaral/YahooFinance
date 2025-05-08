import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------------------
# Part 1: Simulate Daily Returns & Partial Summation
# -------------------------------------

# Set random seed for reproducibility
np.random.seed(42)

# Simulate daily returns for 100 trading days.
# (For realism: mean ~0.1% per day, standard deviation ~2%)
num_days = 100
daily_returns = np.random.normal(loc=0.001, scale=0.02, size=num_days)

# Compute cumulative returns using partial summation (cumulative sum)
cumulative_returns = np.cumsum(daily_returns)

# -------------------------------------
# Part 2: Print Intermediate Values
# -------------------------------------
print("Intermediate Values (First 10 Trading Days):")
print("Day\tDaily Return\tCumulative Return")
for day in range(10):
    print(f"{day+1:3d}\t{daily_returns[day]:.4f}\t\t{cumulative_returns[day]:.4f}")

# -------------------------------------
# Part 3: Plot Static Charts
# -------------------------------------
days = np.arange(1, num_days + 1)

plt.figure(figsize=(12, 5))

# Plot Daily Returns
plt.subplot(1, 2, 1)
plt.plot(days, daily_returns, marker='o', linestyle='-', color='dodgerblue')
plt.title("Daily Returns")
plt.xlabel("Trading Day")
plt.ylabel("Daily Return")
plt.grid(True)

# Plot Cumulative Returns
plt.subplot(1, 2, 2)
plt.plot(days, cumulative_returns, marker='o', linestyle='-', color='firebrick')
plt.title("Cumulative Returns (Partial Summation)")
plt.xlabel("Trading Day")
plt.ylabel("Cumulative Return")
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------
# Part 4: Animate Cumulative Returns Build-Up
# -------------------------------------

# Create the figure for animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Animated Cumulative Returns Over Time")
ax.set_xlabel("Trading Day")
ax.set_ylabel("Cumulative Return")
ax.set_xlim(1, num_days)
# Set y-axis limits a little beyond the min and max of cumulative returns for clear visibility
ax.set_ylim(np.min(cumulative_returns)*1.1, np.max(cumulative_returns)*1.1)
ax.grid(True)

# Initialize an empty line object to update later
line, = ax.plot([], [], color='firebrick', marker='o', linestyle='-', lw=2)

def init():
    # Initializes the animated line as empty
    line.set_data([], [])
    return line,

def update(frame):
    # For each frame, we want to show days from 1 to frame+1
    current_days = np.arange(1, frame + 2)  # frame+1 points (starting at day 1)
    current_cumret = cumulative_returns[:frame + 1]
    line.set_data(current_days, current_cumret)
    
    # Optionally, annotate the current point with its value for added clarity.
    # We'll clear and add a single annotation each frame:
    ax.collections.clear()  # remove previous annotations if any
    ax.text(current_days[-1], current_cumret[-1],
            f"{current_cumret[-1]:.4f}",
            fontsize=10, color="black", ha="center", va="bottom")
    
    return line,

# Create the animation: interval is in milliseconds, frames will range over trading days.
anim = FuncAnimation(fig, update, frames=num_days, init_func=init, interval=100, blit=True)

plt.show()