import yfinance as yf
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Define a subset of tickers and factors for demonstration
tickers_anim = ['AAPL', 'MSFT', 'GOOGL']
factors_anim = ['Mkt-RF', 'SMB', 'HML']
n_stocks_anim = len(tickers_anim)
n_factors_anim = len(factors_anim)

# Define the date range
start_date_adv = "2020-01-01"
end_date_adv = "2024-01-01"

try:
    # Download stock data (monthly returns)
    data_daily_adv = yf.download(tickers_anim, start=start_date_adv, end=end_date_adv)['Close']
    data_monthly_adv = data_daily_adv.resample('ME').last()
    monthly_returns_adv = data_monthly_adv.pct_change().dropna()

    # Download Fama-French 5-Factor Data
    ff_data_adv = web.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date_adv, end=end_date_adv)[0] / 100
    ff_data_adv.index = ff_data_adv.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [ax.plot([], [], marker='o', linestyle='-', alpha=0.7, label=ticker)[0] for ticker in tickers_anim]
    ax.set_xticks(np.arange(n_factors_anim))
    ax.set_xticklabels(factors_anim)
    ax.set_title('Animated Fama-French Factor Loadings')
    ax.legend()
    ax.grid(True)

    def animate_parallel(i):
        if i < len(ff_data_adv):
            factor_values = ff_data_adv[factors_anim].iloc[i].values
            for idx, ticker in enumerate(tickers_anim):
                # For simplicity, we're just plotting the factor values directly.
                # In a real Fama-French regression, you'd have betas for each factor.
                y_values = factor_values
                ax.set_ylim(min(y_values) - 0.05, max(y_values) + 0.05)
                lines[idx].set_data(np.arange(n_factors_anim), y_values)
            ax.set_xlabel(f"Time: {ff_data_adv.index[i].strftime('%Y-%m')}")
        return lines

    ani_parallel = animation.FuncAnimation(fig, animate_parallel, frames=len(ff_data_adv), interval=200, blit=True)
    plt.show()

except Exception as e:
    print(f"Error during advanced animation: {e}")