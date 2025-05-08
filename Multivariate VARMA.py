import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.varmax import VARMAX
import warnings

# Optionally suppress specific warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# --- Define DJIA tickers (10 Companies) ---
djia_tickers = ['AAPL', 'MSFT', 'JNJ', 'V', 'WMT', 'PG', 'IBM', 'KO', 'DIS', 'MCD']
start_date = "2018-01-01"
end_date   = "2023-01-01"

# --- Download data (using 'Close' prices) ---
data_djia = yf.download(djia_tickers, start=start_date, end=end_date)['Close']

# --- Resample daily data to monthly frequency (use "ME" for Month End) ---
data_djia_m = data_djia.resample('ME').last()

# --- Compute monthly returns ---
returns_djia = data_djia_m.pct_change().dropna()

# --- Fit VARMA Model (using VARMAX) ---
# Here, we use an order (1, 1) for demonstration purposes.
model_varmax = VARMAX(returns_djia, order=(1, 1))
result_varmax = model_varmax.fit(disp=False)

# Forecast next 6 months
forecast_var = result_varmax.get_forecast(steps=6).predicted_mean
forecast_dates = pd.date_range(start=returns_djia.index[-1] + pd.offsets.MonthEnd(),
                               periods=6, freq='ME')

# --- Convert datetime indices to numeric values for 3D plotting ---
# This conversion changes the datetime index to floats so that they can be used in the 3D coordinate arrays.
x_obs = mdates.date2num(returns_djia.index.to_pydatetime())
x_forecast = mdates.date2num(forecast_dates.to_pydatetime())

# --- Create 3D plot for VARMA observed and forecast data ---
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot observed returns for two series "AAPL" (y-axis) and "MSFT" (z-axis)
ax.plot(x_obs, 
        returns_djia['AAPL'].values, 
        returns_djia['MSFT'].values,
        label='Observed', color='blue', lw=2)

# Plot forecast data as scatter points for "AAPL" and "MSFT"
ax.scatter(x_forecast, 
           forecast_var['AAPL'].values, 
           forecast_var['MSFT'].values,
           label='Forecast', color='red', s=50)

# Label axes (Note: x-axis is numeric dates; we format it back into dates below.)
ax.set_xlabel('Date')
ax.set_ylabel('AAPL Return')
ax.set_zlabel('MSFT Return')
ax.legend()

# Format the x-axis to show dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# --- Animate the view in 3D ---
def update_3d(frame):
    ax.view_init(elev=30, azim=frame)
    return ax,

ani = FuncAnimation(fig, update_3d, frames=np.linspace(0, 360, 120),
                    interval=100, blit=False)

plt.title("3D Animation of VARMA Forecast (DJIA: AAPL vs MSFT)")
plt.show()