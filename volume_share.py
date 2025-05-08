import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from scipy.stats import norm
from statsmodels.tsa.statespace.varmax import VARMAX  # (unused here but included from earlier examples)
import warnings

# Optionally suppress warnings (you can remove these lines if you want to see them)
warnings.filterwarnings("ignore", category=FutureWarning)

############################################################
# PART I – DATA DOWNLOAD, FIT, AND SORTING

# Define 10 companies (feel free to adjust the list)
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'IBM', 'ORCL', 'CSCO', 'INTC']
start_date = "2018-01-01"
end_date   = "2023-01-01"

# Download daily Close prices for all the companies
data_all = yf.download(companies, start=start_date, end=end_date)['Close']

# Resample daily data to monthly using Month End frequency ('ME')
data_monthly = data_all.resample('ME').last()

# Compute monthly returns
returns = data_monthly.pct_change().dropna()

# For each company, compute:
# - Cumulative return = (product(1 + return) - 1)*100
# - Fitted normal parameters: mu (mean) and sigma (std dev) from monthly returns
# - “Water Volume” computed from the fitted normal PDF by revolving it about the vertical axis.
#   Here we assume a one-dimensional normal PDF (centered at zero) given by:
#       f(x) = 1/(σ√(2π)) exp( -½ (x/σ)² )
#   We define a water volume as:
#       V = 2π ∫₀^(3σ) x f(x) dx  = 2π (σ²/√(2π)) (1 – exp(-4.5))
#   (Note: For a standard normal with μ = 0; if μ ≠ 0 the symmetry breaks, so we use this as a proxy.)
cum_returns = {}
stats_dict = {}

for comp in companies:
    r_series = returns[comp]
    cumulative_return = (np.prod(1 + r_series) - 1) * 100
    mu_val = r_series.mean()
    sigma_val = r_series.std()
    volume = 2 * np.pi * (sigma_val**2) / np.sqrt(2 * np.pi) * (1 - np.exp(-4.5))
    cum_returns[comp] = cumulative_return
    stats_dict[comp] = {'mu': mu_val, 'sigma': sigma_val, 'volume': volume}

print("Intermediate Fitted Parameters and Cumulative Returns:")
for comp in companies:
    print(f"{comp}: μ = {stats_dict[comp]['mu']:.6f}, σ = {stats_dict[comp]['sigma']:.6f}, "
          f"Volume = {stats_dict[comp]['volume']:.6f}, Cumulative Return = {cum_returns[comp]:.2f}%")

# Sort companies by cumulative return descending
sorted_companies = sorted(companies, key=lambda x: cum_returns[x], reverse=True)
print("\nCompanies sorted by cumulative return (descending):")
for comp in sorted_companies:
    print(f"{comp}: {cum_returns[comp]:.2f}%")

############################################################
# PART II – ANIMATION OF THE EVOLUTION OF THE NORMAL PDF (2D)

# For demonstration, choose the company with highest cumulative return.
chosen_company = sorted_companies[0]
print(f"\nAnimating PDF evolution for {chosen_company}")

# Use a rolling window of 12 months on the monthly returns for the chosen company
window = 12
r_chosen = returns[chosen_company]
n_windows = len(r_chosen) - window + 1

# Store rolling parameters and corresponding end time of each window.
time_points = []
rolling_mu = []
rolling_sigma = []
rolling_volume = []

for i in range(n_windows):
    window_data = r_chosen.iloc[i:i+window]
    time_points.append(window_data.index[-1])
    m = window_data.mean()
    s = window_data.std()
    rolling_mu.append(m)
    rolling_sigma.append(s)
    vol = 2 * np.pi * (s**2) / np.sqrt(2 * np.pi) * (1 - np.exp(-4.5))
    rolling_volume.append(vol)

# Prepare a 2D plot showing the fitted normal PDF for each rolling window.
fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5))
x_vals = np.linspace(-0.2, 0.2, 300)  # x axis values (return)
# Initial PDF using the parameters from the first window.
pdf_init = norm.pdf(x_vals, loc=rolling_mu[0], scale=rolling_sigma[0])
line_pdf, = ax_pdf.plot(x_vals, pdf_init, 'b-', lw=2)
time_text = ax_pdf.text(0.05, 0.95, '', transform=ax_pdf.transAxes, fontsize=12, verticalalignment='top')
param_text = ax_pdf.text(0.05, 0.85, '', transform=ax_pdf.transAxes, fontsize=12, verticalalignment='top')
ax_pdf.set_xlabel('Return')
ax_pdf.set_ylabel('Probability Density')
ax_pdf.set_title(f'PDF Evolution for {chosen_company}')

def update_pdf(frame):
    mu_t = rolling_mu[frame]
    sigma_t = rolling_sigma[frame]
    pdf_vals = norm.pdf(x_vals, loc=mu_t, scale=sigma_t)
    line_pdf.set_ydata(pdf_vals)
    t_str = time_points[frame].strftime('%Y-%m')
    time_text.set_text(f"Time: {t_str}")
    param_text.set_text(f"μ = {mu_t:.4f}, σ = {sigma_t:.4f}, Volume = {rolling_volume[frame]:.6f}")
    return line_pdf, time_text, param_text

ani_pdf = FuncAnimation(fig_pdf, update_pdf, frames=n_windows, interval=500, blit=True)
plt.show()

############################################################
# PART III – 3D ANIMATION: "WATER CYLINDER" EVOLUTION

# In this animation the idea is to show a 3D surface of a cylinder whose
# surface is given by revolving the normal PDF (assumed centered at 0) about
# the vertical axis. For simplicity we use only the σ parameter from the rolling window.
# (We assume μ ≈ 0 for the water cylinder; if μ is nonzero, more adjustments are needed.)

def create_cylinder_surface(sigma, num_points=50):
    # Use r in [0, 3σ] (since 3σ roughly covers most mass) and theta [0, 2π].
    r_max = 3 * sigma
    r = np.linspace(0, r_max, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    R, Theta = np.meshgrid(r, theta)
    # Convert polar coordinates to Cartesian for plotting:
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    # The "water surface" height is given by the normal PDF for a random variable ~ N(0,σ²):
    Z = norm.pdf(R, loc=0, scale=sigma)
    return X, Y, Z

# Set up the 3D plot
fig_cyl = plt.figure(figsize=(10, 7))
ax_cyl = fig_cyl.add_subplot(111, projection='3d')

# Initialize the surface. (We store the surface plot object so we can update it.)
surf = None

def update_cylinder(frame):
    global surf
    sigma_t = rolling_sigma[frame]
    X, Y, Z = create_cylinder_surface(sigma_t)
    # Remove the previous surface(s) if they exist.
    for coll in ax_cyl.collections[:]:
        coll.remove()
    # Plot the new surface.
    surf = ax_cyl.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax_cyl.set_title(f"Water Cylinder for {chosen_company} (σ = {sigma_t:.4f})")
    # Set consistent limits – note that the maximum height of the normal PDF is at x=0
    ax_cyl.set_zlim(0, 1 / (min(rolling_sigma)*np.sqrt(2 * np.pi)) * 1.1)
    return surf,

ani_cyl = FuncAnimation(fig_cyl, update_cylinder, frames=n_windows, interval=500, blit=False)
plt.show()