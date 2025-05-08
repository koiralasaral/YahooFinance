import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.dates as mdates
import scipy.optimize as sco
from scipy.stats import norm
import warnings

# Optionally suppress future warnings:
warnings.filterwarnings("ignore", category=FutureWarning)

##############################################################################
# PART A: DATA DOWNLOAD & PREPROCESSING
##############################################################################
# Define 10 companies
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'IBM', 'ORCL', 'CSCO', 'INTC']
start_date = "2018-01-01"
end_date   = "2023-01-01"

print("Downloading monthly closing prices for companies:")
print(companies)

# Download daily closing prices and select 'Close'
data_all = yf.download(companies, start=start_date, end=end_date)['Close']

# Resample to monthly values using Month End frequency ('ME')
data_monthly = data_all.resample('ME').last().dropna()
print("\nSample of downloaded monthly data (first 5 rows):")
print(data_monthly.head())

# Compute monthly returns
returns = data_monthly.pct_change().dropna()

##############################################################################
# PART B: MARKOWITZ MEAN–VARIANCE OPTIMIZATION
##############################################################################
# Use the monthly returns for our 10 companies as assets.
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_assets = len(companies)

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

def min_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def constraint_sum(weights):
    return np.sum(weights) - 1

def constraint_return(weights, target, mean_returns):
    return np.dot(weights, mean_returns) - target

bounds = tuple((0, 1) for _ in range(num_assets))
initial_guess = num_assets * [1. / num_assets]

# Create grid for target returns: create 50 points between the min and max asset mean returns.
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
efficient_portfolios = []  # each element: [target_return, portfolio_volatility, weights]

print("\n=== Markowitz MVO Optimization Intermediate Values ===")
for target in target_returns:
    constraints = ( 
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'eq', 'fun': lambda w, t=target: constraint_return(w, t, mean_returns)}
    )
    result = sco.minimize(min_variance, initial_guess, args=(cov_matrix.values,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        port_return, port_vol = portfolio_performance(result.x, mean_returns, cov_matrix)
        efficient_portfolios.append([port_return, port_vol, result.x])
        print(f"Target = {target:.4f}, Achieved Return = {port_return:.4f}, Volatility = {port_vol:.4f}")
    else:
        efficient_portfolios.append([np.nan, np.nan, None])

# Create an array containing only the portfolio return and volatility to form a homogeneous array.
efficient_frontier = np.array([[p[0], p[1]] for p in efficient_portfolios if not np.isnan(p[0])])
print("\nEfficient Frontier Points (Return, Volatility):")
print(efficient_frontier)

# Animate the efficient frontier: gradually reveal the frontier points.
fig_front, ax_front = plt.subplots(figsize=(8, 6))
ax_front.set_xlabel("Portfolio Volatility")
ax_front.set_ylabel("Portfolio Return")
ax_front.set_title("Markowitz Efficient Frontier")
(line_front,) = ax_front.plot([], [], 'bo-', lw=2)

def init_front():
    line_front.set_data([], [])
    return line_front,

def update_front(i):
    pts = efficient_frontier[:i+1]
    vols = pts[:,1]
    rets = pts[:,0]
    line_front.set_data(vols, rets)
    return line_front,

ani_front = animation.FuncAnimation(fig_front, update_front, frames=len(efficient_frontier),
                                    init_func=init_front, interval=150, blit=True)
plt.show()

##############################################################################
# PART C: BLACK–LITTERMAN OPTIMIZATION AND ANIMATION
##############################################################################
delta = 2.5
tau = 0.025
w_mkt = np.array([1/num_assets]*num_assets)
pi = delta * cov_matrix.dot(w_mkt)

# Define one view: "AAPL outperforms MSFT by 2%".
P = np.zeros((1, num_assets))
idx_AAPL = companies.index("AAPL")
idx_MSFT = companies.index("MSFT")
P[0, idx_AAPL] = 1
P[0, idx_MSFT] = -1
q = np.array([0.02])

# Vary the view uncertainty (omega) from 0.1 (low confidence) to 0.001 (high confidence).
omegas = np.linspace(0.1, 0.001, 50)

def black_litterman_returns(tau, sigma, pi, P, omega, q):
    Omega = np.array([[omega]])
    inv_tauSigma = np.linalg.inv(tau * sigma)
    M = np.linalg.inv(inv_tauSigma + P.T.dot(np.linalg.inv(Omega)).dot(P))
    r_BL = M.dot(inv_tauSigma.dot(pi) + P.T.dot(np.linalg.inv(Omega)).dot(q))
    return r_BL

bl_weights = []
print("\n=== Black–Litterman Optimization Intermediate Values ===")
for omega in omegas:
    r_BL = black_litterman_returns(tau, cov_matrix.values, pi.values, P, omega, q)
    # Optimal weights: w* = (1/δ)*Σ⁻¹ r_BL, then normalize to sum to 1.
    weights = np.linalg.inv(cov_matrix.values).dot(r_BL) / delta
    weights = weights / np.sum(weights)
    bl_weights.append(weights)
    print(f"omega={omega:.4f} -> BL return: {r_BL.flatten()}, weights: {weights.flatten()}")

bl_weights = np.array(bl_weights)
# Animate the evolution of the weight for AAPL vs. omega.
fig_bl, ax_bl = plt.subplots(figsize=(8, 5))
ax_bl.set_xlabel("View Uncertainty (omega)")
ax_bl.set_ylabel("AAPL Weight")
ax_bl.set_title("Evolution of AAPL Weight in Black–Litterman Portfolios")
ax_bl.set_xlim(omegas.min(), omegas.max())
ax_bl.set_ylim(0, 1)
(line_bl,) = ax_bl.plot([], [], 'b-o', lw=2)

def init_bl():
    line_bl.set_data([], [])
    return line_bl,

def update_bl(i):
    x_data = omegas[:i+1]
    y_data = bl_weights[:i+1, idx_AAPL]
    line_bl.set_data(x_data, y_data)
    return line_bl,

ani_bl = animation.FuncAnimation(fig_bl, update_bl, frames=len(omegas),
                                 init_func=init_bl, interval=150, blit=True)
plt.show()