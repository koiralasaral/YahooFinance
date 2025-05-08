import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.varmax import VARMAX
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# --- Define DJIA Tickers (10 Companies) ---
djia_tickers = ['AAPL', 'MSFT', 'JNJ', 'V', 'WMT', 'PG', 'IBM', 'KO', 'DIS', 'MCD']
start_date = "2018-01-01"
end_date   = "2023-01-01"

# Download the data (Close prices)
data_djia = yf.download(djia_tickers, start=start_date, end=end_date)['Close']

# Resample daily data to monthly using Month End frequency ('ME')
data_djia_m = data_djia.resample('ME').last()

# Compute monthly returns
returns_djia = data_djia_m.pct_change().dropna()

# --- Fit VARMA Model using VARMAX with order (1,1) ---
model_varmax = VARMAX(returns_djia, order=(1, 1))
result_varmax = model_varmax.fit(disp=False)

# Forecast the next 6 months
forecast_var = result_varmax.get_forecast(steps=6).predicted_mean
forecast_dates = pd.date_range(start=returns_djia.index[-1] + pd.offsets.MonthEnd(),
                               periods=6, freq='ME')

# --- Convert Dates to Numeric for 3D Plotting ---
x_obs = mdates.date2num(returns_djia.index.to_pydatetime())
x_forecast = mdates.date2num(forecast_dates.to_pydatetime())

# --- Create a 3D Plot for the VARMA Forecast ---
# For illustration, we use 'AAPL' and 'MSFT' series.
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Plot observed returns: x (dates), y (AAPL returns), z (MSFT returns)
ax.plot(x_obs, returns_djia['AAPL'].values, returns_djia['MSFT'].values,
        label='Observed', color='blue', lw=2)
# Plot forecast points for AAPL and MSFT as scatter
ax.scatter(x_forecast, forecast_var['AAPL'].values, forecast_var['MSFT'].values,
           label='Forecast', color='red', s=50)

ax.set_xlabel('Date')
ax.set_ylabel('AAPL Return')
ax.set_zlabel('MSFT Return')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

def update_3d(frame):
    ax.view_init(elev=30, azim=frame)
    return ax,

ani = FuncAnimation(fig, update_3d, frames=np.linspace(0, 360, 120),
                    interval=100, blit=False)
plt.title("3D Animation of VARMA Forecast (DJIA: AAPL vs MSFT)")
plt.show()

# --- MGF Animation for VARMA Case ---
# Animating the MGF of N(0, sigma^2) as sigma increases.
t_vals = np.linspace(-3, 3, 400)
mu = 0
sigma_init = 0.5

fig2, ax2 = plt.subplots(figsize=(8,5))
line, = ax2.plot(t_vals, np.exp(mu*t_vals + 0.5*sigma_init**2*t_vals**2), 'b-', lw=2)
txt = ax2.text(0.05, 0.95, f'sigma = {sigma_init:.2f}', transform=ax2.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8))
ax2.set_xlabel('t')
ax2.set_ylabel('MGF(t)')
ax2.set_title('MGF Animation (VARMA Case)')
ax2.set_ylim(0, 50)

def update_mgf(frame):
    sigma = 0.5 + frame*(3 - 0.5) / 100
    y_val = np.exp(mu*t_vals + 0.5*sigma**2*t_vals**2)
    line.set_ydata(y_val)
    txt.set_text(f'sigma = {sigma:.2f}')
    return line, txt

ani2 = FuncAnimation(fig2, update_mgf, frames=100, interval=100, blit=True)
plt.show()