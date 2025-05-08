import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
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

# Download daily closing prices and then choose 'Close'.
data_all = yf.download(companies, start=start_date, end=end_date)['Close']
# Resample to monthly using Month End frequency ('ME')
data_monthly = data_all.resample('ME').last().dropna()
print("\nSample of downloaded monthly data:")
print(data_monthly.head())

# Compute monthly returns.
returns = data_monthly.pct_change().dropna()

# For each company, compute the expanding (cumulative) average of log returns and build nested intervals.
all_nested_intervals = {}  # company -> list of nested intervals I_n
all_sequence = {}          # company -> sequence x_n
for comp in companies:
    series = data_monthly[comp]
    log_ret = np.log(series / series.shift(1)).dropna()
    cum_avg = log_ret.expanding().mean()
    x_vals = cum_avg.to_numpy().flatten()
    all_sequence[comp] = x_vals
    
    nested_intervals = []
    current_ints = []
    print(f"\n=== {comp} ===")
    for i, xi in enumerate(x_vals):
        print(f"n = {i+1:2d} : x_{i+1} = {float(xi):.6f}")
        eps = 1.0 / (i + 2)  # ε_i = 1/(i+2)
        new_int = (xi - eps, xi + eps)
        current_ints.append(new_int)
        lower = max(iv[0] for iv in current_ints)
        upper = min(iv[1] for iv in current_ints)
        inter = (lower, upper)
        nested_intervals.append(inter)
        width = upper - lower
        print(f"  Interval I_{i+1}: new = [{new_int[0]:.6f}, {new_int[1]:.6f}], Intersection = [{lower:.6f}, {upper:.6f}], width = {width:.6f}")
    all_nested_intervals[comp] = nested_intervals
    print("Final Intersection for", comp, ":", nested_intervals[-1])

##############################################################################
# PART II: 3D CYLINDER ANIMATION OF NESTED INTERVALS (with Blue Liquid Fill)
##############################################################################
# In this part, we’ll use each nested interval to form a container.
# The container’s radius is set as half the interval’s width.
# The liquid level is computed as:
#    fill_fraction = 1 - (current_width - final_width) / (initial_width - final_width)
# and the liquid is drawn as a blue filled cylinder inside the container.
# We use a constant container height, H = 0.05.
H = 0.05

# Precompute, for each company, the initial and final nested interval width.
init_width = {}
final_width = {}
max_radius = {}  # to set axis limits for each company subplot.
for comp in companies:
    ints = all_nested_intervals[comp]
    w0 = ints[0][1] - ints[0][0]
    wf = ints[-1][1] - ints[-1][0]
    init_width[comp] = w0
    final_width[comp] = wf
    # Also compute the maximum radius over all frames.
    radii = [(iv[1] - iv[0]) / 2 for iv in ints]
    max_radius[comp] = max(radii)

# Define helper functions for drawing the container and liquid.
def draw_container(ax, R, H, z_offset=0):
    # Draw container surface (lateral wall) using a mesh in polar coordinates.
    num_theta, num_z = 50, 10
    theta = np.linspace(0, 2*np.pi, num_theta)
    z = np.linspace(0, H, num_z)
    Theta, Z = np.meshgrid(theta, z)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z_offset = Z + z_offset
    # Draw with a light grey color and transparency.
    surf = ax.plot_surface(X, Y, Z_offset, color='gray', alpha=0.3, edgecolor='none')
    return surf

def draw_liquid(ax, R, H, fill_height, z_offset=0):
    # fill_height is the height to which the liquid fills (0 <= fill_height <= H).
    num_theta, num_z = 50, 10
    theta = np.linspace(0, 2*np.pi, num_theta)
    z = np.linspace(0, fill_height, num_z)
    Theta, Z = np.meshgrid(theta, z)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z_offset = Z + z_offset
    # Draw the blue liquid cylinder with less transparency.
    surf = ax.plot_surface(X, Y, Z_offset, color='blue', alpha=0.8, edgecolor='none')
    return surf

# Determine the minimum number of frames (nested intervals) over all companies.
min_frames = min(len(all_nested_intervals[comp]) for comp in companies)
print(f"\nAnimating {min_frames} frames for all companies (Cylinder with Liquid Fill).")

# Create a grid of 3D subplots: 2 rows x 5 columns.
fig_cyl, axes = plt.subplots(nrows=2, ncols=5, subplot_kw={'projection': '3d'}, figsize=(20,10))
axes = axes.flatten()

# Initialize each subplot: set title and fixed axis limits based on max radius.
for i, comp in enumerate(companies):
    ax = axes[i]
    ax.set_title(comp)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    R_max = max_radius[comp]
    ax.set_xlim(-R_max*1.5, R_max*1.5)
    ax.set_ylim(-R_max*1.5, R_max*1.5)
    ax.set_zlim(0, H)
    
def update_cylinders(frame):
    for i, comp in enumerate(companies):
        ax = axes[i]
        # Clear previous drawings.
        ax.cla()
        # Reset axis limits and titles.
        R_max = max_radius[comp]
        ax.set_xlim(-R_max*1.5, R_max*1.5)
        ax.set_ylim(-R_max*1.5, R_max*1.5)
        ax.set_zlim(0, H)
        ax.set_title(comp + f" (Frame {frame+1})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height")
        # Get the nested interval for this frame for company comp.
        interval = all_nested_intervals[comp][frame]
        width = interval[1] - interval[0]
        R = width / 2  # container radius.
        # Compute fill fraction:
        w0 = init_width[comp]
        wf = final_width[comp]
        if w0 != wf:
            fill_fraction = 1 - (width - wf) / (w0 - wf)
            # Ensure fill_fraction is between 0 and 1 (it should be).
            fill_fraction = max(0, min(fill_fraction, 1))
        else:
            fill_fraction = 1.0
        fill_height = fill_fraction * H
        # Draw the container (always drawn from z=0 to H).
        draw_container(ax, R, H, z_offset=0)
        # Draw the liquid fill (from z=0 to fill_height).
        draw_liquid(ax, R, H, fill_height, z_offset=0)
        # Annotate the current width and fill level.
        ax.text(0, 0, H*0.9, f"w={width:.6f}\nlevel={fill_height:.6f}", color="red", fontsize=10, ha='center')
    return axes

ani_cyl = FuncAnimation(fig_cyl, update_cylinders, frames=min_frames, interval=500, blit=False)
plt.suptitle("3D Cylinder Animation of Nested Intervals with Liquid Fill", fontsize=16)
plt.show()

##############################################################################
# PART III: MARKOWITZ MEAN–VARIANCE OPTIMIZATION AND ANIMATION (for 10 companies)
##############################################################################
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
initial_guess = num_assets * [1./num_assets]

target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
efficient_portfolios = []
print("\n--- Markowitz MVO Optimization Intermediate Values ---")
for target in target_returns:
    constraints = [{'type': 'eq', 'fun': constraint_sum},
                   {'type': 'eq', 'fun': lambda w, t=target: constraint_return(w, t, mean_returns)}]
    result = sco.minimize(min_variance, initial_guess, args=(cov_matrix.values,), 
                          method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        port_return, port_vol = portfolio_performance(result.x, mean_returns, cov_matrix)
        efficient_portfolios.append([port_return, port_vol, result.x])
        print(f"Target = {target:.4f}, Achieved Return = {port_return:.4f}, Volatility = {port_vol:.4f}")
    else:
        efficient_portfolios.append([np.nan, np.nan, None])
# Build an array for (return, volatility) pairs.
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
# PART IV: BLACK–LITTERMAN OPTIMIZATION AND ANIMATION (for 10 companies)
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