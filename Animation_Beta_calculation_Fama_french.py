import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define 10 S&P 500 tickers
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JPM', 'V', 'NVDA', 'JNJ']

# Define the date range for analysis
start_date = "2020-01-01"
end_date = "2024-01-01"
rolling_window = 36  # 3-year rolling window for beta estimation

# Download monthly stock data
print("Downloading monthly stock data...")
data_monthly = yf.download(tickers, start=start_date, end=end_date)['Adj Close'].resample('ME').last().pct_change().dropna()

# Download Fama-French 5-Factor Data
print("\nDownloading Fama-French 5-Factor Data...")
ff_data = web.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date, end=end_date)[0] / 100
ff_data.index = ff_data.index.to_timestamp()

# Ensure both datasets have the same frequency and aligned dates
ff_data.index = ff_data.index.normalize() + pd.offsets.MonthEnd(0)
data_monthly.index = data_monthly.index.normalize() + pd.offsets.MonthEnd(0)

# Initialize a dictionary to store the rolling betas
rolling_betas = {ticker: pd.DataFrame() for ticker in tickers}

print("\nCalculating rolling Fama-French betas...")
for ticker in tickers:
    print(f"Calculating for {ticker}...")
    ticker_returns = data_monthly[ticker]
    merged_data = pd.merge(ticker_returns, ff_data, left_index=True, right_index=True, how='inner').dropna()
    if not merged_data.empty:
        exog_vars = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        betas_list = []
        dates_list = []
        for i in range(rolling_window, len(merged_data)):
            y = merged_data[ticker][i - rolling_window:i] - merged_data['RF'][i - rolling_window:i]
            X = sm.add_constant(merged_data[exog_vars][i - rolling_window:i])
            model = sm.OLS(y, X).fit()
            betas_list.append(model.params[exog_vars].tolist())
            dates_list.append(merged_data.index[i])

        beta_df = pd.DataFrame(betas_list, index=dates_list, columns=exog_vars)
        rolling_betas[ticker] = beta_df
    else:
        print(f"No overlapping data for {ticker} and Fama-French factors.")

print("\nCreating Plotly visualization...")
fig = make_subplots(rows=len(tickers), cols=1, subplot_titles=tickers, shared_xaxes=True)

for i, ticker in enumerate(tickers):
    if not rolling_betas[ticker].empty:
        for factor in rolling_betas[ticker].columns:
            fig.add_trace(go.Scatter(x=rolling_betas[ticker].index, y=rolling_betas[ticker][factor],
                                     mode='lines', name=factor, showlegend=(i == 0)),
                          row=i + 1, col=1)
    else:
        fig.add_annotation(text="No beta data available", xref="paper", yref="paper",
                             x=0.5, y=0.5 + (0.9 / len(tickers)) * i, showarrow=False,
                             font=dict(size=12), row=i + 1, col=1)

fig.update_layout(title='Rolling Fama-French Beta Coefficients for 10 Stocks',
                  height=400 * len(tickers),
                  yaxis_title='Beta Coefficient',
                  xaxis_title='Time')
fig.show()