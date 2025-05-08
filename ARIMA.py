import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web

# --------------------
# 1. Define a list of 30 S&P 500 tickers.
tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JNJ', 'V', 'JPM', 'NVDA',
    'PG', 'DIS', 'MA', 'HD', 'BAC', 'VZ', 'ADBE', 'CMCSA', 'NFLX', 'INTC',
    'T', 'XOM', 'KO', 'PFE', 'MRK', 'CVX', 'WMT', 'ABT', 'CRM', 'LLY'
]

# --------------------
# 2. Download daily adjusted close price data.
print("Downloading daily stock data...")
data_daily = yf.download(tickers, start="2020-01-01", end="2025-01-01")['Close']
print("\nHead of Daily Stock Data:")
print(data_daily.head())

# --------------------
# 3. Resample daily data to monthly frequency (using last observation of each month).
# (A FutureWarning will appear: you can replace 'M' with 'ME' if preferred.)
data_monthly = data_daily.resample('M').last()
print("\nHead of Monthly Stock Data:")
print(data_monthly.head())

# --------------------
# 4. Compute monthly returns for each stock.
monthly_returns = data_monthly.pct_change().dropna()
print("\nHead of Monthly Returns:")
print(monthly_returns.head())

# --------------------
# 5. Download Fama–French 5-Factor data.
print("\nDownloading Fama–French 5-Factor Data...")
ff_factors = web.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', 
                                     start='2020-01-01', end='2025-01-01')[0]

# Convert the PeriodIndex to a DatetimeIndex with normalized time.
# This forces the factor dates to have a time of 00:00:00 (midnight).
ff_factors.index = pd.to_datetime(ff_factors.index.strftime('%Y-%m-%d'))

# Convert percentages to decimals.
ff_factors = ff_factors / 100
print("\nHead of Fama–French Factors Data:")
print(ff_factors.head())

# --------------------
# 6. For each stock, merge its monthly returns with the FF factors,
# compute excess returns (Return - RF), and run the regression.
results = {}   # To store cumulative profit percentages
alphas = {}    # To store regression intercepts (alpha)

print("\nRunning Fama–French regressions for each stock:")
for ticker in tickers:
    # Extract the stock's monthly return series and merge with the factors by date.
    stock_return = monthly_returns[ticker].dropna()
    df = pd.merge(stock_return.to_frame(name='Return'),
                  ff_factors,
                  left_index=True, right_index=True,
                  how='inner')
    
    # Check if there is overlapping data.
    if df.empty:
        print(f"\nNo overlapping data for {ticker}. Skipping regression.")
        continue
    
    # Compute excess return: stock monthly return minus RF.
    df['ExcessReturn'] = df['Return'] - df['RF']
    
    # Prepare independent variables (the five factors) and add a constant.
    X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X)
    y = df['ExcessReturn']
    
    # Run the OLS regression.
    model = sm.OLS(y, X).fit()
    alpha = model.params['const']
    alphas[ticker] = alpha
    
    # Compute cumulative profit percentage:
    # (Product of (1 + monthly return)) - 1, then expressed as a percentage.
    cum_profit_pct = (np.prod(1 + df['Return']) - 1) * 100
    results[ticker] = cum_profit_pct
    
    # Print results.
    print(f"\nRegression results for {ticker}:")
    print(model.summary().tables[1])
    print(f"Cumulative Profit Percentage for {ticker}: {cum_profit_pct:.2f}%")

# --------------------
# 7. Create and sort a results DataFrame by cumulative profit percentage.
results_df = pd.DataFrame({
    'CumulativeProfitPct': pd.Series(results),
    'Alpha': pd.Series(alphas)
})
results_df = results_df.sort_values(by='CumulativeProfitPct', ascending=False)

print("\nSorted Results (Descending by Cumulative Profit Percentage):")
print(results_df)

# --------------------
# 8. Plot the cumulative profit percentages.
plt.figure(figsize=(14, 7))
sns.barplot(x=results_df.index, y=results_df['CumulativeProfitPct'], palette="viridis")
plt.ylabel("Cumulative Profit (%)")
plt.xlabel("Stocks")
plt.title("Cumulative Profit Percentage of 30 S&P 500 Stocks (2020-2025)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

#--- Download NVDA Data and Prepare Monthly Series ---
ticker = "NVDA"
start_date = "2018-01-01"
end_date   = "2023-01-01"
data = yf.download(ticker, start=start_date, end=end_date)['Close']

# Resample to monthly frequency (last observation in month)
data_monthly = data.resample('M').last()
series = data_monthly.dropna()

#--- Fit ARIMA Model ---
# We use ARIMA(1,1,1) as an example.
model = ARIMA(series, order=(1, 1, 1))
fit_model = model.fit()

# Forecast the next 6 months
forecast = fit_model.forecast(steps=6)
forecast_dates = pd.date_range(start=series.index[-1] + pd.offsets.MonthEnd(),
                               periods=6, freq='M')

#--- 3D Animation of ARIMA Forecast ---
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot observed series on plane z=0
ax.plot(series.index, series, zs=0, zdir='z', label='Observed', color='blue', lw=2)
# Plot forecast on plane z=1
ax.plot(forecast_dates, forecast, zs=1, zdir='z', label='Forecast', color='red', lw=2, linestyle='--')

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_zlabel('Series Type\n(0: Observed, 1: Forecast)')
ax.legend()

# Rotate the view for 3D animation.
def update(frame):
    ax.view_init(elev=30, azim=frame)
    return ax,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 120), interval=100, blit=False)
plt.title("3D Animation of ARIMA Forecast for NVDA")
plt.show()

#--- MGF Animation for ARIMA Case ---
# Here we animate the moment generating function (MGF) of a Normal distribution:
# MGF(t) = exp(mu*t + 0.5 * sigma^2 * t^2). We fix mu=0 and vary sigma from 0.5 to 3.
t_values = np.linspace(-3, 3, 400)
mu = 0

fig2, ax2 = plt.subplots(figsize=(8,5))
sigma_initial = 0.5
line_mgf, = ax2.plot(t_values, np.exp(mu*t_values + 0.5 * sigma_initial**2 * t_values**2),
                     'b-', lw=2)
text = ax2.text(0.05, 0.95, f'sigma = {sigma_initial:.2f}', transform=ax2.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax2.set_xlabel('t')
ax2.set_ylabel('MGF(t)')
ax2.set_title('MGF Animation (ARIMA Case)')
ax2.set_ylim(0, 50)

def update_mgf(frame):
    sigma = 0.5 + frame*(3 - 0.5)/100
    y_vals = np.exp(mu*t_values + 0.5 * sigma**2 * t_values**2)
    line_mgf.set_ydata(y_vals)
    text.set_text(f'sigma = {sigma:.2f}')
    return line_mgf, text

ani2 = FuncAnimation(fig2, update_mgf, frames=100, interval=100, blit=True)
plt.show()