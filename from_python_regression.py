import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web

# Define a list of 30 S&P 500 tickers.
tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JNJ', 'V', 'JPM', 'NVDA',
    'PG', 'DIS', 'MA', 'HD', 'BAC', 'VZ', 'ADBE', 'CMCSA', 'NFLX', 'INTC',
    'T', 'XOM', 'KO', 'PFE', 'MRK', 'CVX', 'WMT', 'ABT', 'CRM', 'LLY'
]

# Define your date range.
start_date = "2020-01-01"
end_date = "2024-01-01"  # Adjust to ensure data availability

print("Downloading daily stock data...")
data_daily = yf.download(tickers, start=start_date, end=end_date)['Close']

if data_daily.empty:
    print("No stock data retrieved. Check your connection, tickers, or date range.")
    exit()

print("\nHead of Daily Stock Data:")
print(data_daily.head())

# Resample daily data to monthly using Month End ('ME').
data_monthly = data_daily.resample('ME').last()
print("\nHead of Monthly Stock Data:")
print(data_monthly.head())

# Compute monthly returns.
monthly_returns = data_monthly.pct_change()
print("\nHead of Monthly Returns:")
print(monthly_returns.head())

# Download Fama–French 5-Factor Data.
print("\nDownloading Fama–French 5-Factor Data...")
ff_data = web.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date, end=end_date)
ff_factors = ff_data[0]

# Convert the PeriodIndex to Timestamp format and ensure end-of-month
ff_factors.index = ff_factors.index.to_timestamp()
ff_factors.index = ff_factors.index.normalize() + pd.offsets.MonthEnd(0)
print("\nHead of Fama–French Factors Data:")
print(ff_factors.head())
print("\nFama-French Factors Date Range:")
print(f"Start: {ff_factors.index.min()}, End: {ff_factors.index.max()}")

# Convert percentages to decimals.
ff_factors = ff_factors / 100

# Initialize dictionaries for storing results.
results = {}   # Cumulative profit percentages
alphas = {}     # Regression intercepts

print("\nRunning Fama–French regressions for each stock:")
for ticker in tickers:
    # Get stock's monthly return series.
    stock_return = monthly_returns[ticker]

    # Ensure stock return index is also at the end of the month
    stock_return.index = stock_return.index.normalize() + pd.offsets.MonthEnd(0)

    # Merge the stock returns with Fama–French factor data.
    df = pd.merge(stock_return.to_frame(name='Return'),
                    ff_factors,
                    left_index=True, right_index=True,
                    how='left')

    # Check for NaN in Fama-French factors
    if df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].isnull().any().any():
        print(f"\nWarning: NaN values found in Fama-French factors for {ticker}. Handling by dropping rows.")
        df = df.dropna(subset=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
        if df.empty:
            print(f"No data left for {ticker} after dropping NaN Fama-French factors. Skipping.")
            continue

    # Compute the excess return: Return minus the risk-free rate.
    df['ExcessReturn'] = df['Return'] - df['RF']

    # Prepare independent variables and add a constant.
    X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X)
    y = df['ExcessReturn']

    # Run OLS regression.
    try:
        model = sm.OLS(y, X).fit()
        alpha = model.params['const']
        alphas[ticker] = alpha

        # Compute cumulative profit percentage.
        cum_profit_pct = (np.prod(1 + df['Return']) - 1) * 100
        results[ticker] = cum_profit_pct

        # Print regression details and cumulative profit.
        print(f"\nRegression results for {ticker}:")
        print(model.summary().tables[1])
        print(f"Cumulative Profit Percentage for {ticker}: {cum_profit_pct:.2f}%")
    except Exception as e:
        print(f"\nError running regression for {ticker}: {e}")
        continue

# Create and sort the results DataFrame.
results_df = pd.DataFrame({
    'CumulativeProfitPct': pd.Series(results),
    'Alpha': pd.Series(alphas)
})
results_df = results_df.sort_values(by='CumulativeProfitPct', ascending=False)

print("\nSorted Results (Descending by Cumulative Profit Percentage):")
print(results_df)

# Plot the cumulative profit percentages.
plt.figure(figsize=(14, 7))
sns.barplot(x=results_df.index, y=results_df['CumulativeProfitPct'], palette="viridis")
plt.ylabel("Cumulative Profit (%)")
plt.xlabel("Stocks")
plt.title("Cumulative Profit Percentage of 30 S&P 500 Stocks (2020-2024)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("cumulative_profit_percentage.png")
plt.show()