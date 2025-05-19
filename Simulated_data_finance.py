import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from mpl_toolkits.mplot3d import Axes3D
import yfinance as yf

# =============================================================================
# 1. Download Real Data via Yahoo Finance for 12 Defense Stocks
# =============================================================================
# Define defense tickers: Lockheed Martin, Boeing, and 10 additional names.
tickers = ["LMT", "BA", "NOC", "RTX", "GD", "LHX", "HII", "LDOS", "KTOS", "TXT", "BAH", "MRCY"]

# Analysis period: January 1, 2025 to May 20, 2025.
start_date = "2025-01-01"
end_date   = "2025-05-20"

stock_data = {}
print("Downloading historical data...")
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        print(f"Warning: No data available for {ticker}.")
        continue
    df.dropna(inplace=True)
    # Compute key time-series: Daily Return, Cumulative Return and Cumulative Volume.
    df['Daily Return'] = df['Close'].pct_change()
    df['Cumulative Return'] = df['Close'] / df['Close'].iloc[0] - 1
    df['Cumulative Volume'] = df['Volume'].cumsum()
    stock_data[ticker] = df

# =============================================================================
# 2. Fetch Fundamental Information for Each Stock
# =============================================================================
fundamentals = {}
print("\nFetching fundamental data...")
for ticker in tickers:
    try:
        info = yf.Ticker(ticker).info
        fundamentals[ticker] = {
            "EPS": info.get("trailingEps", np.nan),
            "PE Ratio": info.get("trailingPE", np.nan),
            "Dividend Payout Ratio": info.get("payoutRatio", np.nan)
        }
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        fundamentals[ticker] = {"EPS": np.nan, "PE Ratio": np.nan, "Dividend Payout Ratio": np.nan}

# =============================================================================
# 3. Combined 3D Scatter Plot for All Stocks:
#     (Daily Return vs. Volume vs. Price)
# =============================================================================
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for ticker, df in stock_data.items():
    # Use available datapoints – note: drop the initial NaN in Daily Return.
    returns = df['Daily Return'].dropna().values
    vol     = df['Volume'].values
    price   = df['Close'].values
    ax.scatter(returns, vol, price, label=ticker, s=20)
ax.set_xlabel("Daily Return")
ax.set_ylabel("Volume")
ax.set_zlabel("Price")
ax.set_title("Combined 3D Scatter: Daily Return vs Volume vs Price")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# =============================================================================
# Helper: Piecewise Function Fit
# =============================================================================
def piecewise_fit(time_arr, y_arr, split_index):
    """
    Fits a piecewise function:
      - For indices < split_index: a linear (degree 1) fit.
      - For indices >= split_index: a quadratic (degree 2) fit.
      
    Returns:
      - f_piecewise: a function that, given time t, evaluates the appropriate piece,
      - coef1: the coefficients of the first segment,
      - coef2: the coefficients of the second segment.
    """
    coef1 = np.polyfit(time_arr[:split_index], y_arr[:split_index], 1)
    coef2 = np.polyfit(time_arr[split_index:], y_arr[split_index:], 2)
    f_piecewise = lambda t: np.where(t < split_index,
                                     np.polyval(coef1, t),
                                     np.polyval(coef2, t))
    return f_piecewise, coef1, coef2

# =============================================================================
# 4. Analysis for Each Stock: Piecewise Functions, Weierstrass M-test,
#    Differentiation/Integration, and Multivariable Modeling
# =============================================================================
for ticker, df in stock_data.items():
    print("\n==========================================")
    print(f"Analysis for: {ticker}")
    print("==========================================")
    
    # --- Retrieve Fundamental Values (or use 0 if not available)
    eps_val = fundamentals[ticker]["EPS"] if not np.isnan(fundamentals[ticker]["EPS"]) else 0.0
    pe_val  = fundamentals[ticker]["PE Ratio"] if not np.isnan(fundamentals[ticker]["PE Ratio"]) else 0.0
    dp_val  = fundamentals[ticker]["Dividend Payout Ratio"] if not np.isnan(fundamentals[ticker]["Dividend Payout Ratio"]) else 0.0
    
    # Create a time array (in days) from the index of df.
    t_arr = np.arange(len(df))
    dates = df.index
    
    # --- Prepare Variables for Piecewise Modeling ---
    price       = df['Close'].values
    eps_series  = np.full_like(price, eps_val, dtype=float)
    pe_series   = np.full_like(price, pe_val, dtype=float)
    dp_series   = np.full_like(price, dp_val, dtype=float)
    cum_return  = df['Cumulative Return'].values
    cum_volume  = df['Cumulative Volume'].values
    
    variables = {
        "Price": price,
        "EPS": eps_series,
        "PE Ratio": pe_series,
        "Dividend Payout Ratio": dp_series,
        "Cumulative Return": cum_return,
        "Cumulative Volume": cum_volume
    }
    
    # --- 4. Piecewise Function Fitting ---
    n_vars = len(variables)
    fig, axs = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
    split_index = len(t_arr) // 2
    print(f"Piecewise Function Definitions for {ticker}:")
    for i, (var_name, y_data) in enumerate(variables.items()):
        f_piecewise, coef1, coef2 = piecewise_fit(t_arr, y_data, split_index)
        y_fit = f_piecewise(t_arr)
        domain_str = f"[{dates[0].date()} to {dates[-1].date()}]"
        range_min = np.min(y_fit)
        range_max = np.max(y_fit)
        print(f"- {var_name}: Domain: {domain_str}, Range (approx): [{range_min:.4f}, {range_max:.4f}]")
        # Convert coefficients to float before formatting
        print(f"  For t < {split_index}: y(t) = ({float(coef1[0]):.4f})*t + ({float(coef1[1]):.4f})")
        print(f"  For t >= {split_index}: y(t) = ({float(coef2[0]):.4e})*t² + ({float(coef2[1]):.4e})*t + ({float(coef2[2]):.4e})\n")
        ax = axs[i]
        ax.plot(dates, y_data, 'o', markersize=4, label="Data")
        ax.plot(dates, y_fit, '-', linewidth=2, color='red', label="Piecewise Fit")
        ax.set_ylabel(var_name)
        ax.legend(loc='upper left')
        ax.grid(True)
    axs[-1].set_xlabel("Date")
    plt.suptitle(f"Piecewise Fit Models for {ticker}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # --- 5. Weierstrass M-Test Demonstration ---
    daily_returns = df['Daily Return'].dropna().values
    N = len(daily_returns)
    x_space = np.linspace(0, 2*np.pi, 200)
    k_values = [min(10, N), min(20, N), min(50, N), N]
    partial_sums = []
    for k in k_values:
        S = np.zeros_like(x_space)
        for n in range(min(k, N)):
            S += daily_returns[n] * np.cos(x_space) / ((n+1)**2)
        partial_sums.append(S)
    plt.figure(figsize=(10, 6))
    for j, k in enumerate(k_values):
        plt.plot(x_space, partial_sums[j], label=f"Partial Sum k = {k}")
    plt.xlabel("x")
    plt.ylabel("Sₖ(x)")
    plt.title(f"Weierstrass M-Test Partial Sums for {ticker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Also plot the cumulative sum of bounds
    M_n = np.abs(daily_returns) / ((np.arange(1, N+1))**2)
    upper_bound = np.cumsum(M_n)
    plt.figure(figsize=(10, 5))
    plt.plot(upper_bound, 'o-', label='Cumulative Upper Bound Sum')
    plt.xlabel("n")
    plt.ylabel("Upper Bound Sum")
    plt.title(f"Upper Bound Sum for {ticker} (Weierstrass M-Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # --- 6. Differentiation, Gradient, and Integration on Cumulative Return ---
    y = cum_return
    # Fit a 2nd-degree polynomial to the cumulative return.
    coeffs = np.polyfit(t_arr, y, 2)
    poly_func = np.poly1d(coeffs)
    y_fit_poly = poly_func(t_arr)
    dy_dt = np.gradient(y_fit_poly, 1)
    area_under_curve = simps(y_fit_poly, t_arr)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(dates, y, 'o', markersize=4, label="Actual Cumulative Return")
    plt.plot(dates, y_fit_poly, '-', linewidth=2, label="2nd Degree Poly Fit")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"Cumulative Return & Polynomial Fit for {ticker}")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(dates, dy_dt, '-', color='green', label="Derivative (Gradient)")
    plt.xlabel("Date")
    plt.ylabel("d(Return)/dt")
    plt.title("Gradient of Cumulative Return")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print(f"Cumulative area under curve for {ticker} (integration): {area_under_curve:.4f}")
    
    # --- 7. Multivariable Real Analysis (3D Surface & Gradient Field) ---
    # Model Daily Return as a function of time (t) and traded volume (v)
    t_model = np.arange(len(df['Daily Return'].dropna()))
    vol_model = df['Volume'].values[:len(t_model)]
    ret_model = df['Daily Return'].dropna().values
    A = np.column_stack((
         np.ones(len(t_model)),
         t_model,
         vol_model,
         t_model * vol_model,
         t_model**2,
         vol_model**2
    ))
    coeffs_poly, residuals, rank, s = np.linalg.lstsq(A, ret_model, rcond=None)
    print("Fitted multivariate polynomial for Daily Return (f(t,v)):")
    print("f(t, v) = {:.4e} + {:.4e}*t + {:.4e}*v + {:.4e}*t*v + {:.4e}*t² + {:.4e}*v²".format(
          float(coeffs_poly[0]), float(coeffs_poly[1]), float(coeffs_poly[2]),
          float(coeffs_poly[3]), float(coeffs_poly[4]), float(coeffs_poly[5])
    ))
    if residuals.size > 0:
        print("Residual sum of squares:", residuals[0])
    else:
        print("Residual sum of squares not available.")
    
    fitted_func = lambda t, v: (coeffs_poly[0] +
                                coeffs_poly[1]*t +
                                coeffs_poly[2]*v +
                                coeffs_poly[3]*t*v +
                                coeffs_poly[4]*t**2 +
                                coeffs_poly[5]*v**2)
    
    t_min_model, t_max_model = np.min(t_model), np.max(t_model)
    v_min_model, v_max_model = np.min(vol_model), np.max(vol_model)
    T_grid, V_grid = np.meshgrid(np.linspace(t_min_model, t_max_model, 50),
                                 np.linspace(v_min_model, v_max_model, 50))
    F_grid = fitted_func(T_grid, V_grid)
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(T_grid, V_grid, F_grid, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(f"3D Surface for {ticker}\nFitted f(t, v) for Daily Return")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Daily Traded Volume")
    ax.set_zlabel("Daily Return")
    
    dT = T_grid[0, 1] - T_grid[0, 0]
    dV = V_grid[1, 0] - V_grid[0, 0]
    grad_V, grad_T = np.gradient(F_grid, dV, dT)
    
    ax2 = fig.add_subplot(122)
    q = ax2.quiver(T_grid, V_grid, grad_T, grad_V, color='teal')
    ax2.set_title(f"Gradient Field for {ticker}")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Daily Traded Volume")
    
    plt.tight_layout()
    plt.show()
    
print("\nComplete analysis for all stocks.")