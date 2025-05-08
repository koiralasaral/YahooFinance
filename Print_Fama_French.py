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

# Suppress FutureWarnings for clarity.
warnings.filterwarnings("ignore", category=FutureWarning)

##############################################################################
# PART I: DATA DOWNLOAD AND NESTED INTERVAL SEQUENCE COMPUTATION
##############################################################################
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'IBM', 'ORCL', 'CSCO', 'INTC']
start_date = "2018-01-01"
end_date   = "2023-01-01"

print("Downloading monthly closing prices for companies:")
print(companies)

# Download daily closing prices and select 'Close'
data_all = yf.download(companies, start=start_date, end=end_date)['Close']
# Resample to monthly data using Month End frequency ('ME')
data_monthly = data_all.resample('ME').last().dropna()
print("\nSample of downloaded monthly data:")
print(data_monthly.head())

# Compute monthly returns
returns = data_monthly.pct_change().dropna()

# For each company, compute the expanding average of log returns and nested intervals.
all_nested_intervals = {}  # Map: company => list of nested intervals I_n
all_sequence = {}          # Map: company => sequence x_n
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
        # Print the cumulative average.
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
# PART II: 3D CYLINDER ANIMATION OF NESTED INTERVALS WITH LIQUID FILL
##############################################################################
# We use each nested interval to define a container.
# Container: radius = (width/2) and fixed height H.
# The liquid fill level is computed from the current width:
#    fill_fraction = 1 - (current_width - final_width) / (initial_width - final_width)
# (if initial!=final; otherwise fill_fraction = 1).
H = 0.05  # fixed container height

# Precompute initial and final widths for each company.
init_width = {}
final_width = {}
max_radius = {}  # maximum radius (width/2) over frames, used for axis limits.
for comp in companies:
    ints = all_nested_intervals[comp]
    w0 = ints[0][1] - ints[0][0]
    wf = ints[-1][1] - ints[-1][0]
    init_width[comp] = w0
    final_width[comp] = wf
    radii = [(iv[1] - iv[0]) / 2 for iv in ints]
    max_radius[comp] = max(radii)

# Helper functions to draw container and liquid.
def draw_container(ax, R, H, z_offset=0):
    # Create cylinder surface using polar coordinates.
    num_theta, num_z = 50, 10
    theta = np.linspace(0, 2*np.pi, num_theta)
    z = np.linspace(0, H, num_z)
    Theta, Z = np.meshgrid(theta, z)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z_offset = Z + z_offset
    # Draw the container wall (transparent gray).
    surf = ax.plot_surface(X, Y, Z_offset, color='gray', alpha=0.3, edgecolor='none')
    return surf

def draw_liquid(ax, R, H, fill_height, z_offset=0):
    # Draw a blue-filled cylinder up to fill_height.
    num_theta, num_z = 50, 10
    theta = np.linspace(0, 2*np.pi, num_theta)
    z = np.linspace(0, fill_height, num_z)
    Theta, Z = np.meshgrid(theta, z)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z_offset = Z + z_offset
    surf = ax.plot_surface(X, Y, Z_offset, color='blue', alpha=0.8, edgecolor='none')
    return surf

# Determine minimum frames over all companies.
min_frames = min(len(all_nested_intervals[comp]) for comp in companies)
print(f"\nAnimating {min_frames} frames for all companies (Cylinder with Liquid Fill).")

# Create a grid of 3D subplots for 10 companies (2 rows x 5 columns).
fig_cyl, axes = plt.subplots(nrows=2, ncols=5, subplot_kw={'projection':'3d'}, figsize=(20,10))
axes = axes.flatten()

# Initialize each subplot.
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
        ax.cla()  # Clear current draw.
        R_max = max_radius[comp]
        ax.set_xlim(-R_max*1.5, R_max*1.5)
        ax.set_ylim(-R_max*1.5, R_max*1.5)
        ax.set_zlim(0, H)
        ax.set_title(comp + f" (Frame {frame+1})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height")
        # Get current nested interval.
        interval = all_nested_intervals[comp][frame]
        width = interval[1] - interval[0]
        R = width / 2
        # Compute fill fraction.
        w0 = init_width[comp]
        wf = final_width[comp]
        if w0 != wf:
            fill_fraction = 1 - ((width - wf) / (w0 - wf))
            fill_fraction = max(0, min(fill_fraction, 1))
        else:
            fill_fraction = 1.0
        fill_height = fill_fraction * H
        # Draw container and then liquid fill.
        draw_container(ax, R, H, z_offset=0)
        draw_liquid(ax, R, H, fill_height, z_offset=0)
        ax.text(0, 0, H*0.9, f"w={width:.6f}\nLiquidLvl={fill_height:.6f}", color="red", fontsize=10, ha='center')
    return axes

ani_cyl = FuncAnimation(fig_cyl, update_cylinders, frames=min_frames, interval=500, blit=False)
plt.suptitle("3D Cylinder Animation with Blue Liquid (Nested Intervals)", fontsize=16)
plt.show()

##############################################################################
# PART III: MARKOWITZ MEAN–VARIANCE OPTIMIZATION (MVO) AND EFFICIENT FRONTIER
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
# Efficient frontier as an array (columns: return, volatility).
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

# Print explanation of the efficient frontier:
print("\n=== Efficient Frontier Explanation ===")
print("The efficient frontier is a curve representing the set of portfolios that achieve the highest expected return")
print("for a given level of risk (volatility) or, equivalently, the lowest risk for a given return.")
print("Portfolios on the left have lower volatility and lower returns, while those on the right have higher volatility")
print("and higher expected returns. Based on your risk tolerance, you might choose a portfolio in a specific segment")
print("of the frontier. For example, a risk-averse investor may choose a portfolio from the left end, sacrificing some return")
print("for lower risk, whereas a more aggressive investor may choose a portfolio with higher expected return despite higher volatility.")

##############################################################################
# PART IV: BLACK–LITTERMAN OPTIMIZATION WITH 10 VIEWS AND ANIMATION
##############################################################################
delta = 2.5
tau = 0.025
w_mkt = np.array([1/num_assets]*num_assets)
pi = delta * cov_matrix.dot(w_mkt)

# Set 10 views—one for each stock. Use P as the 10x10 identity.
P = np.eye(num_assets)
# Investor-defined views (excess returns) for each stock:
# For example (in decimal form):
# AAPL: +2%, MSFT: +1.5%, GOOGL: +1.8%, AMZN: +2.2%, META: +1.2%,
# NVDA: +2.5%, IBM: -0.5%, ORCL: +0.5%, CSCO: +1.0%, INTC: -1.0%
q = np.array([0.02, 0.015, 0.018, 0.022, 0.012, 0.025, -0.005, 0.005, 0.010, -0.010])
# Vary view uncertainty omega (applied equally to all 10 views)
omegas = np.linspace(0.1, 0.001, 50)
bl_weights = []

print("\n--- Black–Litterman Optimization Intermediate Values (10 Views) ---")
for omega in omegas:
    Omega = omega * np.eye(num_assets)
    inv_tauSigma = np.linalg.inv(tau * cov_matrix.values)
    # BL formula with P=I:
    M = np.linalg.inv(inv_tauSigma + np.linalg.inv(Omega))
    r_BL = M.dot(inv_tauSigma.dot(pi.values) + np.linalg.inv(Omega).dot(q))
    # Optimal weights: w* = (1/δ)*Σ⁻¹ r_BL, then normalized.
    w_BL = np.linalg.inv(cov_matrix.values).dot(r_BL) / delta
    w_BL = w_BL / np.sum(w_BL)
    bl_weights.append(w_BL)
    print(f"omega={omega:.4f} -> BL returns: {r_BL.flatten()}, weights: {w_BL.flatten()}")
bl_weights = np.array(bl_weights)

##############################################################################
# Animate the evolution of portfolio weights for all stocks under Black–Litterman.
##############################################################################
fig_bl, ax_bl = plt.subplots(figsize=(10,6))
ax_bl.set_xlabel("View Uncertainty (omega)")
ax_bl.set_ylabel("Portfolio Weight")
ax_bl.set_title("Evolution of Portfolio Weights (Black–Litterman Model, 10 Views)")
ax_bl.set_xlim(omegas.min(), omegas.max())
ax_bl.set_ylim(0, 1)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
lines = []
for j, comp in enumerate(companies):
    line, = ax_bl.plot([], [], label=comp, color=colors[j], marker='o')
    lines.append(line)
ax_bl.legend()

def init_bl():
    for line in lines:
        line.set_data([], [])
    return lines

def update_bl(i):
    # For each asset, plot the evolution up to frame i.
    for j in range(num_assets):
        x_data = omegas[:i+1]
        y_data = bl_weights[:i+1, j]
        lines[j].set_data(x_data, y_data)
    return lines

ani_bl = animation.FuncAnimation(fig_bl, update_bl, frames=len(omegas),
                                 init_func=init_bl, interval=150, blit=True)
plt.show()

##############################################################################
# FINAL OUTPUT: DECISIONS BASED ON MODELS
##############################################################################
print("\n=== Model Implications and Decision Guidance ===\n")
print("Markowitz MVO Optimization Implications:")
print(" - The efficient frontier represents the set of optimal portfolios for different risk tolerances.")
print(" - Portfolios with lower volatility have lower expected returns, while those with higher volatility")
print("   offer higher expected returns. Your choice depends on your risk appetite.")
print(" - For risk-averse investors, a portfolio on the left side of the frontier may be preferred,")
print("   while aggressive investors may target the right side for higher expected gains.")

print("\nBlack–Litterman Model Implications (with 10 Views):")
print(" - The Black–Litterman model adjusts the equilibrium returns by incorporating investor views for each stock.")
print(" - Here, with 10 views, we have modified the expected returns based on your beliefs (e.g., AAPL +2%, MSFT +1.5%, etc.).")
print(" - As the view uncertainty (omega) decreases, the investor's views have a larger influence;")
print("   this is reflected in the changing portfolio weights.")
print(" - If, for example, the weight for AAPL rises substantially as omega decreases,")
print("   it suggests that your strong positive view on AAPL is influential in the optimal allocation.")
print(" - Decision Suggestion:")
print("   * If you are risk-averse, consider a portfolio near the left portion of the efficient frontier.")
print("   * If you trust your views, the Black–Litterman model suggests overweighting stocks with high positive views (e.g., AAPL, NVDA)")
print("   * You may decide to tilt your portfolio toward the stocks that consistently receive higher weights under lower uncertainty.")
print("\nBased on these models, you should choose a portfolio allocation that balances historical performance (as per MVO)")
print("with your own expectations (as reflected in the BL model). Adjust the weights for stocks like AAPL or NVDA if")
print("their BL weights are significantly higher than their equilibrium weights, provided you accept the associated risk.")

print("\n--- End of Analysis ---")