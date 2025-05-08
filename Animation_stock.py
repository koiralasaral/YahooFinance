import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# =============================================================================
# Part 1: Download real stock data for 20 companies
# =============================================================================

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "UNH", 
    "DIS", "ADBE", "NFLX", "INTC", "MRK"
]

# Dictionary to hold computed metrics for each company.
metrics = {}

# Loop over each ticker to download data and compute returns and moments.
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    try:
        # Download one year of daily data.
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        continue

    if "Adj Close" not in data.columns:
        print(f"Data for {ticker} not available.")
        continue

    # Keep and rename the adjusted close column.
    data = data[['Adj Close']].rename(columns={'Adj Close': 'Adj_Close'}).dropna()
    if data.empty:
        continue

    # =============================================================================
    # Part 2: Compute Daily Returns & Cumulative Return
    # =============================================================================

    # Calculate daily percentage return.
    data['Daily_Return'] = data['Adj_Close'].pct_change().fillna(0)
    # For cumulative return, compute the product of (1 + daily_return) and subtract 1.
    cum_return = np.prod(1 + data['Daily_Return']) - 1

    # Use daily returns as the sample for moment calculations.
    X = data['Daily_Return'].to_numpy()
    
    # =============================================================================
    # Part 3: Compute Moments and Functions
    # =============================================================================
    
    # Raw (origin) moments: E[X^n] for n=1,2,3,4.
    m1 = np.mean(X)
    m2 = np.mean(X**2)
    m3 = np.mean(X**3)
    m4 = np.mean(X**4)
    
    # Central moments about the mean:
    variance = np.mean((X - m1)**2)
    skewness = np.mean((X - m1)**3) / (variance**1.5) if variance > 0 else np.nan
    kurtosis = np.mean((X - m1)**4) / (variance**2) if variance > 0 else np.nan  # Note: kurtosis of a normal = 3
    
    # Define the range for the parameter t (for MGF and CF).
    t_vals = np.linspace(-0.5, 0.5, 200)
    # Moment Generating Function: M_X(t) = E[e^(tX)]
    mgf_vals = np.array([np.mean(np.exp(t * X)) for t in t_vals])
    # Characteristic Function: φ_X(t) = E[e^(i * tX)]
    cf_vals = np.array([np.mean(np.exp(1j * t * X)) for t in t_vals])
    
    # Distribution classification based on skewness and kurtosis.
    # (Here a simple rule: if |skewness| < 0.5 and |kurtosis - 3| < 0.5, we call it "Normal")
    distribution = "Normal" if (abs(skewness) < 0.5 and abs(kurtosis - 3) < 0.5) else "Non-normal"
    
    # Save computed metrics.
    metrics[ticker] = {
        "cumulative_return": cum_return,
        "raw_moments": (m1, m2, m3, m4),
        "central_moments": (variance, skewness, kurtosis),
        "mgf": mgf_vals,
        "cf": cf_vals,
        "t_vals": t_vals,
        "distribution": distribution
    }

# =============================================================================
# Part 4: Create and Print Summary Statistics
# =============================================================================

summary_list = []
for ticker in tickers:
    if ticker in metrics:
        d = metrics[ticker]
        summary_list.append({
            "Ticker": ticker,
            "Cumulative Return": d["cumulative_return"],
            "Mean": d["raw_moments"][0],
            "Variance": d["central_moments"][0],
            "Skewness": d["central_moments"][1],
            "Kurtosis": d["central_moments"][2],
            "Distribution": d["distribution"]
        })
        
summary_df = pd.DataFrame(summary_list)
print("\nSummary Statistics for Daily Returns:")
print(summary_df)

# =============================================================================
# Part 5: Plot MGF and Characteristic Function for Each Stock
# =============================================================================

# Create a grid of subplots: one subplot per stock.
ncols = 4
nrows = int(np.ceil(len(tickers) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    if ticker not in metrics:
        continue
    ax = axes[i]
    t_vals = metrics[ticker]["t_vals"]
    mgf_vals = metrics[ticker]["mgf"]
    cf_vals = metrics[ticker]["cf"]
    
    # Plot the MGF (always real for real-valued X)
    ax.plot(t_vals, mgf_vals, label="MGF", color="blue")
    # Plot the real part of the characteristic function
    ax.plot(t_vals, np.real(cf_vals), label="CF (real)", color="red", linestyle="--")
    # Plot the imaginary part of the characteristic function
    ax.plot(t_vals, np.imag(cf_vals), label="CF (imag)", color="green", linestyle=":")
    
    ax.set_title(f"{ticker} - {metrics[ticker]['distribution']}")
    ax.set_xlabel("t")
    ax.set_ylabel("Function Value")
    ax.legend(fontsize=8)
    
# Hide any unused subplots.
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
    
plt.tight_layout()
plt.show()

# =============================================================================
# Part 6: Animate the Moment Generating Function (MGF) for All Stocks
# =============================================================================

fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(tickers)))
lines = {}

# Create an empty line for each ticker.
for i, ticker in enumerate(tickers):
    if ticker in metrics:
        t_vals = metrics[ticker]["t_vals"]
        # Preallocate an empty line with a distinct color.
        line, = ax_anim.plot([], [], label=ticker, color=colors[i])
        lines[ticker] = line

ax_anim.set_xlim(np.min(t_vals), np.max(t_vals))
# Determine the global ylim across all stocks for proper scaling.
all_mgf = np.concatenate([metrics[t]["mgf"] for t in tickers if t in metrics])
ax_anim.set_ylim(np.min(all_mgf) * 0.9, np.max(all_mgf) * 1.1)
ax_anim.set_title("Animated Moment Generating Function (MGF) for 20 Stocks")
ax_anim.set_xlabel("t")
ax_anim.set_ylabel("MGF")
ax_anim.legend(fontsize=8, loc="upper left")

def animate(frame):
    # For each ticker, update the line with data up to the current frame.
    for ticker in tickers:
        if ticker in metrics:
            t_arr = metrics[ticker]["t_vals"]
            mgf_arr = metrics[ticker]["mgf"]
            idx = min(frame, len(t_arr) - 1)
            lines[ticker].set_data(t_arr[:idx+1], mgf_arr[:idx+1])
    return list(lines.values())

anim_mgf = FuncAnimation(fig_anim, animate, frames=len(t_vals), interval=100, blit=True)
plt.show()

# =============================================================================
# Part 7: 3D Animation for the Stock with the Highest Cumulative Return
# =============================================================================

# Identify the stock with the highest cumulative return.
max_ticker = max(metrics.keys(), key=lambda x: metrics[x]["cumulative_return"])
best = metrics[max_ticker]
print(f"\nStock with Highest Cumulative Return: {max_ticker}")
print(f"Cumulative Return: {best['cumulative_return']:.4f}")

# For the chosen stock, we demonstrate a 3D animation of its characteristic function.
# Since the MGF is real, the CF (being complex) is more interesting for 3D representation.
t_vals = best["t_vals"]
cf_vals = best["cf"]

fig_3d = plt.figure(figsize=(10, 8))
ax3d = fig_3d.add_subplot(111, projection='3d')

# Plot the 3D curve: x-axis: t, y-axis: Re(CF), z-axis: Im(CF)
line3d, = ax3d.plot(t_vals, np.real(cf_vals), np.imag(cf_vals), color='purple', lw=2, marker='o')
ax3d.set_xlabel("t")
ax3d.set_ylabel("Re(CF)")
ax3d.set_zlabel("Im(CF)")
ax3d.set_title(f"3D Characteristic Function for {max_ticker}")

def update_3d(angle):
    ax3d.view_init(elev=30, azim=angle)
    return fig_3d,

anim3d = FuncAnimation(fig_3d, update_3d, frames=np.arange(0, 360, 2), interval=50, blit=False)
plt.show()