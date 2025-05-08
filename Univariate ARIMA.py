import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

# --- Download NVDA Data ---
ticker = "NVDA"
start_date = "2018-01-01"
end_date   = "2023-01-01"
data = yf.download(ticker, start=start_date, end=end_date)['Close']

# Resample daily data to monthly using Month End (ME)
data_monthly = data.resample('ME').last()
series = data_monthly.dropna()

# --- Fit ARIMA(1,1,1) Model ---
model = ARIMA(series, order=(1, 1, 1))
fit_model = model.fit()

# Forecast the next 6 months
forecast = fit_model.forecast(steps=6)
forecast_dates = pd.date_range(start=series.index[-1] + pd.offsets.MonthEnd(),
                               periods=6, freq='ME')

# --- Convert Dates to Numeric Values ---
x_obs = mdates.date2num(series.index.to_pydatetime())
x_forecast = mdates.date2num(forecast_dates.to_pydatetime())

# --- Create 3D Plot for ARIMA Forecast ---
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Plot observed data on plane z=0 (dates converted to numbers)
ax.plot(x_obs, series.values, zs=0, zdir='z', label='Observed', color='blue', lw=2)
# Plot forecast data on plane z=1
ax.plot(x_forecast, forecast.values, zs=1, zdir='z', label='Forecast', color='red', lw=2, linestyle='--')

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_zlabel('Series Type\n(0: Observed, 1: Forecast)')
ax.legend()

# Format the x-axis to display dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

def update(frame):
    ax.view_init(elev=30, azim=frame)
    return ax,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 120), interval=100, blit=False)
plt.title("3D Animation of ARIMA Forecast for NVDA")
plt.show()

# --- MGF Animation for ARIMA Case ---
# Here we animate the moment generating function of N(0, sigma^2)
t_values = np.linspace(-3, 3, 400)
mu = 0
sigma_initial = 0.5

fig2, ax2 = plt.subplots(figsize=(8,5))
line_mgf, = ax2.plot(t_values, np.exp(mu*t_values + 0.5 * sigma_initial**2 * t_values**2),
                      'b-', lw=2)
text = ax2.text(0.05, 0.95, f'sigma = {sigma_initial:.2f}', transform=ax2.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))

ax2.set_xlabel('t')
ax2.set_ylabel('MGF(t)')
ax2.set_title('MGF Animation (ARIMA Case)')
ax2.set_ylim(0, 50)

def update_mgf(frame):
    sigma = 0.5 + frame*(3 - 0.5) / 100
    y_vals = np.exp(mu*t_values + 0.5 * sigma**2 * t_values**2)
    line_mgf.set_ydata(y_vals)
    text.set_text(f'sigma = {sigma:.2f}')
    return line_mgf, text

ani2 = FuncAnimation(fig2, update_mgf, frames=100, interval=100, blit=True)
plt.show()