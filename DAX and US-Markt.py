import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from tqdm import tqdm # For displaying progress bars

# --- 1. Define MDAX and SDAX Stock Tickers ---
# These lists are illustrative and may not be perfectly up-to-date.
# For a real-world application, you would need a reliable source
# to fetch the current constituents of these indices.
MDAX_tickers = [
    "AIR.DE", "ALV.DE", "BAS.DE", "BMW.DE", "CON.DE", "DAI.DE", "DB1.DE", "DHL.DE", 
    "DPW.DE", "DTE.DE", "EON.DE", "FRE.DE", "HEN3.DE", "IFX.DE", "LIN.DE", "LHA.DE", 
    "MRK.DE", "MTX.DE", "MUV2.DE", "NDA.DE", "P911.DE", "RWE.DE", "SAP.DE", "SIE.DE", 
    "SRM.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "ADS.DE", "BEI.DE", "DEQ.DE", 
    "DLG.DE", "ENR.DE", "FME.DE", "HIL.DE", "KCO.DE", "LEG.DE", "PAG.DE", "PMQ.DE", 
    "RHM.DE", "SDF.DE", "TUI1.DE", "VOS.DE", "WCH.DE", "XTRA.DE", "UN01.DE", "PFG.DE", 
    "EVK.DE"
]

SDAX_tickers = [
    "1U1.DE", "AOF.DE", "CAN.DE", "CECG.DE", "COKG.DE", "COP.DE", "CWCG.DE", "DBAN.DE", 
    "DEZG.DE", "DRWG.DE", "ELGG.DE", "EUZG.DE", "FIE.DE", "F3CG.DE", "FYB.DE", "GFT.DE", 
    "GLJ.DE", "HAB.DE", "HDDG.DE", "HBH.DE", "HYQ.DE", "ILM.DE", "INHG.DE", "IOS.DE", 
    "JST.DE", "KCOG.DE", "KSBG.DE", "KWS.DE", "LPKG.DE", "MBB.DE", "MLP.DE", "NCH.DE", 
    "NOEJ.DE", "PAT.DE", "PBB.DE", "PNEG.DE", "PSMG.DE", "PVA.DE", "S92.DE", "SAF.DE", 
    "SGCG.DE", "SHA.DE", "SIX2.DE", "SMHN.DE", "STOG.DE", "SZG.DE", "SZU.DE", "TPEG.DE", 
    "VBK.DE", "VOS.DE", "WAC.DE", "WAF.DE", "WUWG.DE", "DUE.DE", "STM.DE", "GYC.DE", 
    "KTN.DE", "BVB.DE", "ADNG.DE", "EKT.DE", "PCZ.DE", "BFSA.DE", "ACT1.DE", "DMP.DE", 
    "SPGG.DE", "VH2.DE", "1SXP.DE", "DOU1.DE"
]

# Combine all tickers
all_tickers = MDAX_tickers + SDAX_tickers

# --- Configuration for Data Download and HMM ---
START_DATE = "2020-01-01"
END_DATE = "2024-12-31" # Data up to end of 2024 for historical analysis
N_HIDDEN_STATES = 3 # Number of hidden market regimes (e.g., bear, neutral, bull)
TOP_N_STOCKS = 10   # Number of top stocks to display by profit

# --- Data Collection ---
stock_data = {}        # Stores 'Close' prices for each stock
daily_returns = {}     # Stores daily percentage returns for each stock

print(f"Attempting to download historical data for {len(all_tickers)} stocks from {START_DATE} to {END_DATE}...")
for ticker in tqdm(all_tickers, desc="Downloading data"):
    try:
        # Download historical data from Yahoo Finance
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if not data.empty:
            stock_data[ticker] = data['Close']
            # Calculate daily percentage change and drop the first NaN value
            daily_returns[ticker] = data['Close'].pct_change().dropna()
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Could not download data for {ticker}: {e}")

# --- Filter Stocks with Insufficient Data ---
# We need a reasonable amount of data to train the HMM.
# Let's require at least 2 years of daily data (approx. 252 trading days/year * 2)
min_data_points = 252 * 2 
filtered_tickers = [t for t, r in daily_returns.items() if len(r) >= min_data_points]

if not filtered_tickers:
    print("No stocks with sufficient data to perform HMM analysis. Exiting.")
    exit()

print(f"\nProceeding with {len(filtered_tickers)} stocks that have sufficient data.")

# --- Hidden Markov Model (HMM) Implementation ---
hmm_models = {}             # Stores trained HMM models for each stock
stock_hidden_states = {}    # Stores inferred hidden states for each stock's returns
stock_predicted_returns = {} # Stores predicted daily returns based on HMM states

print("\nTraining HMM for each stock and inferring states...")
for ticker in tqdm(filtered_tickers, desc="Training HMMs"):
    # Reshape returns for HMM (hmmlearn expects 2D array: (n_samples, n_features))
    returns_for_hmm = daily_returns[ticker].values.reshape(-1, 1)
    
    try:
        # Initialize and train a GaussianHMM model.
        # GaussianHMM is suitable for continuous observations like stock returns.
        # n_components: number of hidden states.
        # covariance_type: 'full' allows for a full covariance matrix for each state.
        # n_iter: maximum number of iterations for the EM algorithm.
        model = hmm.GaussianHMM(n_components=N_HIDDEN_STATES, covariance_type="full", n_iter=100, tol=0.01)
        model.fit(returns_for_hmm)
        hmm_models[ticker] = model
        
        # Predict the most likely sequence of hidden states given the observed returns.
        hidden_states = model.predict(returns_for_hmm)
        stock_hidden_states[ticker] = hidden_states

        # Calculate the mean return for each hidden state.
        # These means represent the average daily return when the market is in that specific hidden state.
        state_means = [np.mean(returns_for_hmm[hidden_states == i]) for i in range(N_HIDDEN_STATES)]
        
        # Create a sequence of "predicted" daily returns.
        # For each day, the predicted return is the mean return of the hidden state it was inferred to be in.
        predicted_daily_returns_for_stock = np.array([state_means[state] for state in hidden_states])
        stock_predicted_returns[ticker] = predicted_daily_returns_for_stock

    except Exception as e:
        print(f"Could not train HMM for {ticker}: {e}")

# --- Calculate Cumulative Profit ---
cumulative_profits = {}

print("\nCalculating cumulative profits based on HMM-predicted returns...")
for ticker in filtered_tickers:
    if ticker in stock_predicted_returns:
        # Start with an initial investment (e.g., 100 units of currency)
        initial_investment = 100 
        
        # Calculate the cumulative product of (1 + daily_return).
        # This simulates the growth of the initial investment over time.
        # The '1 +' converts returns into growth factors (e.g., 0.01 return becomes 1.01 factor).
        cumulative_product = np.cumprod(1 + stock_predicted_returns[ticker])
        
        # The final value in cumulative_product represents the total accumulated value.
        # Profit is the final accumulated value minus the initial investment.
        profit = (cumulative_product[-1] * initial_investment) - initial_investment
        cumulative_profits[ticker] = profit

# --- Print Top 10 Stocks by Cumulative Profit ---
# Sort the stocks by their calculated cumulative profit in descending order.
top_10_stocks_profit = sorted(cumulative_profits.items(), key=lambda item: item[1], reverse=True)[:TOP_N_STOCKS]

print(f"\n--- Top {TOP_N_STOCKS} Stocks by Cumulative Profit (based on HMM predictions) ---")
if top_10_stocks_profit:
    for ticker, profit in top_10_stocks_profit:
        print(f"{ticker}: {profit:.2f} units of profit")
else:
    print("No stocks found with calculated cumulative profit.")

# --- Analyze Convergence/Divergence and Plotting ---
print("\n--- Analysis of Convergence/Divergence ---")
print("1. Stock Prices: Generally diverge (tend to follow non-stationary processes, can grow indefinitely).")
print("2. Daily Returns: Tend to be stationary (their statistical properties like mean/variance converge).")
print("3. HMM Hidden States: The HMM model itself converges during training (parameters stabilize).")
print("   The sequence of states continues to evolve, reflecting market regimes.")
print("4. Cumulative Profit: For a consistently profitable strategy, cumulative profit generally diverges (grows to infinity).")
print("   There is no finite limit to wealth accumulation in a growing market.")
print("5. Limit of Highest Number of Days with Positive Daily Return:")
print("   This is not a mathematical limit. Over long periods, the *proportion* of positive days tends to a probability.")
print("   The *count* of positive days can increase with the observation window, but its rate of increase might diminish.")


# --- Plotting Convergence Criteria (Illustrative) ---
# We will plot the cumulative value for the top-performing stock
# and a rolling count of positive return days to illustrate concepts.

if top_10_stocks_profit:
    best_ticker_for_plot = top_10_stocks_profit[0][0]
    print(f"\nPlotting for the top stock: {best_ticker_for_plot}")

    # Plot Cumulative Value (illustrates divergence of profit)
    if best_ticker_for_plot in stock_predicted_returns:
        predicted_returns_to_plot = stock_predicted_returns[best_ticker_for_plot]
        
        initial_investment_plot = 100
        # Calculate the cumulative value over time
        cumulative_value_plot = np.cumprod(1 + predicted_returns_to_plot) * initial_investment_plot
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_value_plot, label=f'Cumulative Value of {best_ticker_for_plot}')
        plt.title(f'Cumulative Value Over Time for {best_ticker_for_plot} (HMM-Predicted Returns)')
        plt.xlabel('Days from Start Date')
        plt.ylabel('Cumulative Value (Starting at 100)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"The cumulative value plot for {best_ticker_for_plot} shows a general upward trend, indicating divergence (growth) of profit, as expected for a profitable strategy.")
        print("It does not converge to a finite value but rather continues to increase over time.")

    # Plot Rolling Count of Positive Returns (illustrates fluctuation, not a strict limit)
    if best_ticker_for_plot in daily_returns:
        # Convert daily returns to a boolean series: True if positive, False otherwise
        daily_positive_returns_series = (daily_returns[best_ticker_for_plot] > 0).astype(int)
        
        # Calculate a rolling sum of positive days over a 30-day window
        rolling_positive_days = daily_positive_returns_series.rolling(window=30).sum()
        
        plt.figure(figsize=(12, 6))
        # Plot the rolling count, aligning the x-axis with the dates
        plt.plot(rolling_positive_days.index, rolling_positive_days, label=f'Rolling 30-Day Positive Returns for {best_ticker_for_plot}')
        plt.title(f'Rolling 30-Day Count of Positive Returns for {best_ticker_for_plot}')
        plt.xlabel('Date')
        plt.ylabel('Number of Positive Days in 30-day Window')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Calculate the overall percentage of positive days
        overall_positive_percentage = np.mean(daily_returns[best_ticker_for_plot] > 0) * 100
        print(f"\nOverall percentage of positive daily returns for {best_ticker_for_plot} over the period: {overall_positive_percentage:.2f}%")
        print("The plot of rolling positive days shows fluctuation. While the *probability* of a positive return on any given day might converge over a very long period, the *count* of positive days within a finite window will vary and doesn't have a strict mathematical 'limit' in the sense of converging to a single number as time progresses indefinitely.")

else:
    print("\nNo top stocks to plot as no sufficient data or HMM training failed for all stocks.")

