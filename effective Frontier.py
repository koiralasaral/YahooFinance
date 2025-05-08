import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import scipy.optimize as sco
from scipy.stats import norm
import matplotlib.animation as animation
import warnings

# Optionally suppress FutureWarnings.
warnings.filterwarnings("ignore", category=FutureWarning)

##############################################################################
# PART I: DATA DOWNLOAD AND NESTED INTERVAL SEQUENCE COMPUTATION
##############################################################################
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'IBM', 'ORCL', 'CSCO', 'INTC']
start_date = "2018-01-01"
end_date   = "2023-01-01"

print("Downloading monthly closing prices for companies:")
print(companies)

# Download daily closing prices and take "Close"
data_all = yf.download(companies, start=start_date, end=end_date)['Close']
# Resample to monthly data using Month End frequency ('ME')
data_monthly = data_all.resample('ME').last().dropna()
print("\nSample of downloaded monthly data:")
print(data_monthly.head())

# Compute monthly returns
returns = data_monthly.pct_change().dropna()

# For each company, we compute the expanding (cumulative) average of log returns and the nested intervals.
all_nested_intervals = {}  # company -> list of nested intervals (one per frame)
all_sequence = {}          # company -> the sequence x_n
for comp in companies:
    series = data_monthly[comp]
    # Compute log returns: r_t = ln(P_t/P_{t-1})
    log_ret = np.log(series / series.shift(1)).dropna()
    cum_avg = log_ret.expanding().mean()
    x_vals = cum_avg.to_numpy().flatten()
    all_sequence[comp] = x_vals
    nested_intervals = []
    current_ints = []
    print(f"\n=== {comp} ===")
    for i, xi in enumerate(x_vals):
        # Print the current cumulative average
        print(f"n = {i+1:2d} : x_{i+1} = {float(xi):.6f}")
        eps = 1.0 / (i + 2)  # ε_i = 1/(i+2)
        new_int = (xi - eps, xi + eps)
        current_ints.append(new_int)
        lower = max(iv[0] for iv in current_ints)
        upper = min(iv[1] for iv in current_ints)
        inter = (lower, upper)
        nested_intervals.append(inter)
        width = upper - lower
        print(f" Interval I_{i+1}: new = [{new_int[0]:.6f}, {new_int[1]:.6f}], Intersection = [{lower:.6f}, {upper:.6f}], width = {width:.6f}")
    all_nested_intervals[comp] = nested_intervals
    print("Final Intersection for", comp, ":", nested_intervals[-1])

##############################################################################
# PART II: 3D CYLINDER ANIMATION OF NESTED INTERVALS FOR EACH STOCK
##############################################################################
# For each company, we will animate a 3D cylinder whose radius equals half the width of the nested interval.
# (The idea: as the nested intervals converge, the cylinder shrinks.)
# We arrange all 10 companies in a grid (2 rows x 5 columns).

# Helper function to create a cylinder mesh (with constant height) given a radius.
def create_cylinder(radius, height=1, num_theta=50, num_z=10):
    theta = np.linspace(0, 2*np.pi, num_theta)
    z = np.linspace(0, height, num_z)
    Theta, Z = np.meshgrid(theta, z)
    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)
    return X, Y, Z

# Determine the minimum number of frames (nested intervals) available over all companies.
min_frames = min(len(all_nested_intervals[comp]) for comp in companies)
print(f"\nAnimating {min_frames} frames for all companies (Cylinder Animation).")

# Create a grid of 3D subplots (2 rows x 5 columns)
fig_cyl, axes = plt.subplots(nrows=2, ncols=5, subplot_kw={'projection':'3d'}, figsize=(20,10))
axes = axes.flatten()

# Prepare each subplot with titles and fixed axes limits.
for i, comp in enumerate(companies):
    ax = axes[i]
    ax.set_title(comp)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Frame")
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(0, min_frames)

# Update function for the animation: For each frame, update each company's subplot.
def update_cylinder(frame):
    for i, comp in enumerate(companies):
        ax = axes[i]
        # Remove any previous surfaces.
        for coll in ax.collections[:]:
            coll.remove()
        # Get the nested interval for the current frame.
        interval = all_nested_intervals[comp][frame]
        width = interval[1] - interval[0]
        # Define cylinder radius as half the width.
        r = width / 2
        # Create a cylinder with fixed height (set here to 1) and then offset vertically by the frame index.
        X, Y, Z = create_cylinder(r, height=1)
        Z = Z + frame  # position the cylinder at z=frame.
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        # Annotate the cylinder with its width.
        ax.text(0, 0, frame, f"{width:.6f}", color="red", fontsize=10)
    return axes

ani_cyl = FuncAnimation(fig_cyl, update_cylinder, frames=min_frames, interval=500, blit=False)
plt.suptitle("3D Cylinder Animation of Nested Intervals for Each Stock")
plt.show()

##############################################################################
# PART III: MARKOWITZ MEAN–VARIANCE OPTIMIZATION (MVO) AND ANIMATION
##############################################################################
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_assets = len(companies)

def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def min_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def constraint_sum(weights):
    return np.sum(weights) - 1

def constraint_return(weights, target, mean_returns):
    return np.dot(weights, mean_returns) - target

bounds = tuple((0, 1) for _ in range(num_assets))
initial_guess = num_assets * [1./num_assets]

target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
efficient_portfolios = []
print("\n--- Markowitz MVO Optimization Intermediate Values ---")
for target in target_returns:
    constraints = [{'type':'eq', 'fun': constraint_sum},
                   {'type':'eq', 'fun': lambda w, t=target: constraint_return(w, t, mean_returns)}]
    result = sco.minimize(min_variance, initial_guess, args=(cov_matrix.values,), method='SLSQP',
                          bounds=bounds, constraints=constraints)
    if result.success:
        port_return, port_vol = portfolio_performance(result.x, mean_returns, cov_matrix)
        efficient_portfolios.append([port_return, port_vol, result.x])
        print(f"Target = {target:.4f}, Achieved Return = {port_return:.4f}, Volatility = {port_vol:.4f}")
    else:
        efficient_portfolios.append([np.nan, np.nan, None])
# Form a homogeneous array for the frontier points (only return and vol)
efficient_frontier = np.array([[p[0], p[1]] for p in efficient_portfolios if p[2] is not None])
print("\nEfficient Frontier Points (Return, Volatility):")
print(efficient_frontier)

fig_front, ax_front = plt.subplots(figsize=(8,6))
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
# PART IV: BLACK–LITTERMAN OPTIMIZATION AND ANIMATION
##############################################################################
delta = 2.5
tau = 0.025
w_mkt = np.array([1/num_assets]*num_assets)
pi = delta * cov_matrix.dot(w_mkt)

# Define one view: "AAPL outperforms MSFT by 2%"
P = np.zeros((1, num_assets))
idx_AAPL = companies.index("AAPL")
idx_MSFT = companies.index("MSFT")
P[0, idx_AAPL] = 1
P[0, idx_MSFT] = -1
q = np.array([0.02])

omegas = np.linspace(0.1, 0.001, 50)

def black_litterman_returns(tau, sigma, pi, P, omega, q):
    Omega = np.array([[omega]])
    inv_tauSigma = np.linalg.inv(tau * sigma)
    M = np.linalg.inv(inv_tauSigma + P.T.dot(np.linalg.inv(Omega)).dot(P))
    r_BL = M.dot(inv_tauSigma.dot(pi) + P.T.dot(np.linalg.inv(Omega)).dot(q))
    return r_BL

bl_weights = []
print("\n--- Black–Litterman Optimization Intermediate Values ---")
for omega in omegas:
    r_BL = black_litterman_returns(tau, cov_matrix.values, pi.values, P, omega, q)
    weights = np.linalg.inv(cov_matrix.values).dot(r_BL) / delta
    weights = weights / np.sum(weights)
    bl_weights.append(weights)
    print(f"omega={omega:.4f} -> BL return: {r_BL.flatten()}, weights: {weights.flatten()}")
bl_weights = np.array(bl_weights)

fig_bl, ax_bl = plt.subplots(figsize=(8,5))
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