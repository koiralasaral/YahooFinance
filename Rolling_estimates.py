import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Define 10 companies (adjust as desired)
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'IBM', 'ORCL', 'CSCO', 'INTC']
start_date = "2018-01-01"
end_date   = "2023-01-01"

# Download daily closing prices
data_all = yf.download(companies, start=start_date, end=end_date)['Close']

# Resample daily data to monthly using Month End frequency ('ME')
data_monthly = data_all.resample('ME').last()

# Compute monthly returns
returns = data_monthly.pct_change().dropna()

# For each company compute overall cumulative return and overall normal fit based on all monthly returns.
cum_returns = {}
stats_dict = {}

for comp in companies:
    r_series = returns[comp]
    cumulative_return = (np.prod(1 + r_series) - 1) * 100
    mu_val = r_series.mean()
    sigma_val = r_series.std()
    # Compute "water volume" via revolving the PDF from 0 to 3σ.
    volume = 2 * np.pi * (sigma_val**2) / np.sqrt(2 * np.pi) * (1 - np.exp(-4.5))
    cum_returns[comp] = cumulative_return
    stats_dict[comp] = {'mu': mu_val, 'sigma': sigma_val, 'volume': volume}

print("\n=== Intermediate Fitted Parameters and Cumulative Returns ===")
for comp in companies:
    print(f"{comp}: μ = {stats_dict[comp]['mu']:.6f}, σ = {stats_dict[comp]['sigma']:.6f}, "
          f"Volume = {stats_dict[comp]['volume']:.6f}, Cumulative Return = {cum_returns[comp]:.2f}%")

# Sort companies by cumulative return (descending)
sorted_companies = sorted(companies, key=lambda x: cum_returns[x], reverse=True)
print("\n=== Companies Sorted by Cumulative Return (Descending) ===")
for comp in sorted_companies:
    print(f"{comp}: {cum_returns[comp]:.2f}%")

# --- Rolling window estimates (rolling 12-month window) for each company.
rolling_params = {}  # For each company, store rolling arrays.
window = 12
for comp in companies:
    r_series = returns[comp]
    n_windows = len(r_series) - window + 1
    time_points = []
    rolling_mu = []
    rolling_sigma = []
    rolling_volume = []
    for i in range(n_windows):
        window_data = r_series.iloc[i:i+window]
        time_points.append(window_data.index[-1])
        m = window_data.mean()
        s = window_data.std()
        rolling_mu.append(m)
        rolling_sigma.append(s)
        vol = 2 * np.pi * (s**2) / np.sqrt(2 * np.pi) * (1 - np.exp(-4.5))
        rolling_volume.append(vol)
    rolling_params[comp] = {'time': time_points, 'mu': rolling_mu,
                              'sigma': rolling_sigma, 'volume': rolling_volume}
    import scipy.optimize as sco
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Use historical monthly returns from all companies (we assume these are our assets)
asset_returns = returns.copy()
mean_returns = asset_returns.mean()
cov_matrix = asset_returns.cov()
num_assets = len(companies)

# Risk aversion parameter is not needed for frontier calculation – we find portfolios for a range of target returns.
def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

# Objective: minimize portfolio variance for given target return.
def min_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraint functions for optimization
def constraint_sum(weights):
    return np.sum(weights) - 1

def constraint_return(weights, target, mean_returns):
    return np.dot(weights, mean_returns) - target

# Bounds for weights: [0,1] (no short selling)
bounds = tuple((0, 1) for _ in range(num_assets))
initial_guess = num_assets * [1. / num_assets,]

# Create grid for target returns between min and max of the individual mean returns.
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
efficient_portfolios = []  # store [target_return, vol, weights]

print("\n=== Optimization for Markowitz Mean–Variance Frontier ===")
for r in target_returns:
    constraints = ({'type': 'eq', 'fun': constraint_sum},
                   {'type': 'eq', 'fun': lambda w, r=r: constraint_return(w, r, mean_returns)})
    result = sco.minimize(min_variance, initial_guess, args=(cov_matrix,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        port_return, port_vol = portfolio_performance(result.x, mean_returns, cov_matrix)
        efficient_portfolios.append([port_return, port_vol, result.x])
        # Print intermediate values:
        print(f"Target = {r:.4f}, Achieved Return = {port_return:.4f}, Vol = {port_vol:.4f}")
    else:
        efficient_portfolios.append([np.nan, np.nan, None])

efficient_portfolios = np.array(efficient_portfolios)

# Animate the efficient frontier: gradually reveal the frontier points.
fig_front, ax_front = plt.subplots(figsize=(8,6))
ax_front.set_xlabel("Portfolio Volatility")
ax_front.set_ylabel("Portfolio Return")
ax_front.set_title("Markowitz Efficient Frontier")
(scatter_plot,) = ax_front.plot([], [], 'bo-', lw=2)

def init_front():
    scatter_plot.set_data([], [])
    return scatter_plot,

def animate_front(i):
    pts = efficient_portfolios[:i+1]
    vol = pts[:,1]
    ret = pts[:,0]
    scatter_plot.set_data(vol, ret)
    return scatter_plot,

ani_front = animation.FuncAnimation(fig_front, animate_front,
                                    frames=len(efficient_portfolios),
                                    init_func=init_front, interval=150, blit=True)
plt.show()