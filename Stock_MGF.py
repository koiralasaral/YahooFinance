import time
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import sys

# =============================================================================
# PART 1: Download Data for a List of Stocks One by One with Delay
# =============================================================================

# You might want to start with a smaller list to avoid rate limits.
tickers = ["WMT", "PG", "MA", "HD", "UNH", 
    "DIS", "ADBE", "NFLX", "INTC", "MRK"
]

# Dictionary to store data for each ticker
all_data = {}
delay_seconds = 2  # Adjust delay as needed

print("Downloading data one ticker at a time to avoid rate limiting...")
for ticker in tickers:
    try:
        # Download data for a single ticker with auto_adjust and no progress bar.
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            print(f"Data for {ticker} is empty.")
        else:
            all_data[ticker] = df
            print(f"Downloaded data for {ticker}.")
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
    time.sleep(delay_seconds)  # wait between requests

if not all_data:
    print("No data could be downloaded (possibly due to rate limiting).")
    sys.exit("Exiting: Please try again later or reduce the number of tickers.")

# =============================================================================
# PART 2: Compute Daily Returns, Moments, MGF, & CF for Each Stock
# =============================================================================

metrics = {}
t_vals = np.linspace(-0.5, 0.5, 200)

for ticker, df in all_data.items():
    # Use the "Close" price (already auto_adjusted)
    if "Close" not in df.columns:
        print(f"{ticker} does not have 'Close' data.")
        continue
    close_prices = df["Close"]
    daily_returns = close_prices.pct_change().fillna(0)
    X = daily_returns.to_numpy()
    cum_return = np.prod(1 + X) - 1

    # Raw moments (about the origin)
    m1 = np.mean(X)
    m2 = np.mean(X**2)
    m3 = np.mean(X**3)
    m4 = np.mean(X**4)

    # Central moments (about the mean)
    variance = np.mean((X - m1)**2)
    skewness = np.mean((X - m1)**3) / (variance**1.5) if variance > 0 else np.nan
    kurtosis = np.mean((X - m1)**4) / (variance**2) if variance > 0 else np.nan

    mgf_vals = np.array([np.mean(np.exp(t * X)) for t in t_vals])
    cf_vals = np.array([np.mean(np.exp(1j * t * X)) for t in t_vals])

    # A simple heuristic: classify as "Normal" if skewness ~ 0 and kurtosis ~ 3.
    distribution = "Normal" if (abs(skewness) < 0.5 and abs(kurtosis - 3) < 0.5) else "Non-Normal"

    metrics[ticker] = {
        "daily_returns": daily_returns,
        "cumulative_return": cum_return,
        "raw_moments": (m1, m2, m3, m4),
        "central_moments": (variance, skewness, kurtosis),
        "mgf": mgf_vals,
        "cf": cf_vals,
        "t_vals": t_vals,
        "distribution": distribution
    }

# Check that we have some data
if len(metrics) == 0:
    sys.exit("No valid data was downloaded. Exiting.")

# =============================================================================
# PART 3: Print Summary Statistics for Each Stock
# =============================================================================

summary_list = []
for ticker in metrics:
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
# PART 4: Plot Static MGF & Characteristic Function for Each Stock
# =============================================================================

ncols = 4
nrows = int(np.ceil(len(metrics) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
axes = axes.flatten()

for i, ticker in enumerate(metrics):
    ax = axes[i]
    t_arr = metrics[ticker]["t_vals"]
    mgf_arr = metrics[ticker]["mgf"]
    cf_arr = metrics[ticker]["cf"]

    ax.plot(t_arr, mgf_arr, label="MGF", color="blue")
    ax.plot(t_arr, np.real(cf_arr), label="CF (Real)", color="red", linestyle="--")
    ax.plot(t_arr, np.imag(cf_arr), label="CF (Imag)", color="green", linestyle=":")
    ax.set_title(f"{ticker} ({metrics[ticker]['distribution']})")
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    
for j in range(i+1, len(axes)):
    axes[j].axis("off")
    
plt.tight_layout()
plt.show()

# =============================================================================
# PART 5: Animate the MGF for Each Stock (2D Animation)
# =============================================================================

fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(metrics)))
lines = {}
for idx, ticker in enumerate(metrics):
    line, = ax_anim.plot([], [], label=ticker, color=colors[idx])
    lines[ticker] = line

ax_anim.set_xlim(t_vals[0], t_vals[-1])
all_mgf = np.concatenate([metrics[t]["mgf"] for t in metrics])
ax_anim.set_ylim(np.min(all_mgf) * 0.9, np.max(all_mgf) * 1.1)
ax_anim.set_title("Animated MGF for Stocks")
ax_anim.set_xlabel("t")
ax_anim.set_ylabel("MGF")
ax_anim.legend(fontsize=8, loc="upper left")

def animate(frame):
    for ticker in metrics:
        t_arr = metrics[ticker]["t_vals"]
        mgf_arr = metrics[ticker]["mgf"]
        idx = min(frame, len(t_arr)-1)
        lines[ticker].set_data(t_arr[:idx+1], mgf_arr[:idx+1])
    return list(lines.values())

anim_mgf = FuncAnimation(fig_anim, animate, frames=len(t_vals), interval=100, blit=True)
plt.show()

# =============================================================================
# PART 6: 3D Animation of the MGF for the Top 5 Stocks (by Cumulative Return)
# =============================================================================

# Identify the top 5 stocks by cumulative return.
top_five = summary_df.sort_values("Cumulative Return", ascending=False).head(5)
top5_tickers = top_five["Ticker"].tolist()
print("\nTop Five Stocks by Cumulative Return:")
print(top_five)

fig3d = plt.figure(figsize=(12, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
# Create vertical offsets so the curves don't overlap.
offsets = np.linspace(0, 4, len(top5_tickers))

for idx, ticker in enumerate(top5_tickers):
    t_arr = metrics[ticker]["t_vals"]
    mgf_arr = metrics[ticker]["mgf"]
    y_offset = np.full_like(t_arr, offsets[idx])
    ax3d.plot(t_arr, y_offset, mgf_arr, label=ticker, marker='o', lw=2)

ax3d.set_xlabel("t")
ax3d.set_ylabel("Stock Offset")
ax3d.set_zlabel("MGF")
ax3d.set_title("3D MGF for Top 5 Stocks by Cumulative Return")
ax3d.legend()

def update_3d(angle):
    ax3d.view_init(elev=30, azim=angle)
    return fig3d,

anim3d = FuncAnimation(fig3d, update_3d, frames=np.arange(0, 360, 2), interval=50, blit=False)
plt.show()