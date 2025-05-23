import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from tqdm import tqdm # For displaying progress bars
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
# Removed: import pandas_ta as ta # For SMA

# --- 1. Define MDAX and SDAX Stock Tickers ---
# These lists are illustrative and may not be perfectly up-to-date.
# For a real-world application, you would need a reliable source
# to fetch the current constituents of these indices.


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
all_tickers =  SDAX_tickers

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

# --- Helper Function for Cumulative Profit Calculation ---
def calculate_cumulative_profit(returns_series, initial_investment=100):
    """Calculates cumulative profit from a series of daily returns."""
    if returns_series.empty:
        return 0
    cumulative_product = (1 + returns_series).cumprod()
    
    # Ensure the final value is a scalar to prevent ValueError during sorting
    final_value = cumulative_product.iloc[-1]
    if isinstance(final_value, (pd.Series, np.ndarray)):
        final_value = final_value.item() # Get the scalar value from a single-element Series/array
    
    profit = (final_value * initial_investment) - initial_investment
    return float(profit) # Ensure return is always a float scalar

# --- Hidden Markov Model (HMM) Implementation ---
hmm_models = {}             # Stores trained HMM models for each stock
stock_hidden_states = {}    # Stores inferred hidden states for each stock's returns
stock_predicted_returns_hmm = {} # Stores predicted daily returns based on HMM states

print("\nTraining HMM for each stock and inferring states...")
for ticker in tqdm(filtered_tickers, desc="Training HMMs"):
    # Reshape returns for HMM (hmmlearn expects 2D array: (n_samples, n_features))
    returns_for_hmm = daily_returns[ticker].values.reshape(-1, 1)
    
    try:
        # Initialize and train a GaussianHMM model.
        model = hmm.GaussianHMM(n_components=N_HIDDEN_STATES, covariance_type="full", n_iter=100, tol=0.01)
        model.fit(returns_for_hmm)
        hmm_models[ticker] = model
        
        # Predict the most likely sequence of hidden states given the observed returns.
        hidden_states = model.predict(returns_for_hmm)
        stock_hidden_states[ticker] = hidden_states

        # Calculate the mean return for each hidden state.
        state_means = [np.mean(returns_for_hmm[hidden_states == i]) for i in range(N_HIDDEN_STATES)]
        
        # Create a sequence of "predicted" daily returns.
        predicted_daily_returns_for_stock = np.array([state_means[state] for state in hidden_states])
        stock_predicted_returns_hmm[ticker] = pd.Series(predicted_daily_returns_for_stock, index=daily_returns[ticker].index)

    except Exception as e:
        # print(f"Could not train HMM for {ticker}: {e}") # Uncomment for detailed errors
        pass # Suppress frequent HMM errors for brevity in tqdm

# --- Calculate Cumulative Profit for HMM ---
cumulative_profits_hmm = {}

print("\nCalculating cumulative profits based on HMM-predicted returns...")
for ticker in filtered_tickers:
    if ticker in stock_predicted_returns_hmm:
        profit = calculate_cumulative_profit(stock_predicted_returns_hmm[ticker])
        cumulative_profits_hmm[ticker] = profit

# --- Print Top 10 Stocks by Cumulative Profit (HMM) ---
top_10_stocks_profit_hmm = sorted(cumulative_profits_hmm.items(), key=lambda item: item[1], reverse=True)[:TOP_N_STOCKS]

print(f"\n--- Top {TOP_N_STOCKS} Stocks by Cumulative Profit (Hidden Markov Model) ---")
if top_10_stocks_profit_hmm:
    for ticker, profit in top_10_stocks_profit_hmm:
        print(f"{ticker}: {profit:.2f} units of profit")
else:
    print("No stocks found with calculated cumulative profit for HMM.")


# --- Other Statistical Models for Comparison ---

# --- Model 1: ARIMA (Autoregressive Integrated Moving Average) ---
# Simplified approach: Train on 80% of data, predict on 20% for each stock.
# If predicted return > 0, assume we take the actual daily return; otherwise, 0 return.
cumulative_profits_arima = {}
print("\nTraining ARIMA models and calculating profits...")
TRAIN_SPLIT_RATIO = 0.8 # 80% for training, 20% for testing/prediction

for ticker in tqdm(filtered_tickers, desc="Training ARIMA"):
    returns = daily_returns[ticker]
    
    # Ensure enough data for split and ARIMA fitting
    if len(returns) < 10: # ARIMA needs at least a few points for training and prediction
        continue

    train_size = int(len(returns) * TRAIN_SPLIT_RATIO)
    train_data, test_data = returns[:train_size], returns[train_size:]

    if test_data.empty: # Check if test_data is empty after split
        continue 

    try:
        # Fit ARIMA(1,0,1) model (p=1, d=0, q=1)
        # d=0 because we are modeling returns, which are often considered stationary.
        model = ARIMA(train_data, order=(1,0,1))
        model_fit = model.fit()
        
        # Forecast on the test set (out-of-sample prediction)
        predictions = model_fit.predict(start=len(train_data), end=len(returns)-1)
        
        # Align predictions with test_data index for proper comparison
        predictions.index = test_data.index

        # Strategy: If predicted return is positive, assume we capture the actual return.
        # Otherwise (prediction is non-positive), assume no position (0 return).
        strategy_returns = test_data.copy()
        strategy_returns[predictions <= 0] = 0 

        profit = calculate_cumulative_profit(strategy_returns)
        cumulative_profits_arima[ticker] = profit

    except Exception as e:
        # print(f"Could not train ARIMA for {ticker}: {e}") # Uncomment for detailed errors
        pass # Suppress frequent ARIMA errors for brevity in tqdm output

# --- Print Top 10 Stocks by Cumulative Profit (ARIMA) ---
top_10_stocks_profit_arima = sorted(cumulative_profits_arima.items(), key=lambda item: item[1], reverse=True)[:TOP_N_STOCKS]

print(f"\n--- Top {TOP_N_STOCKS} Stocks by Cumulative Profit (ARIMA Model) ---")
if top_10_stocks_profit_arima:
    for ticker, profit in top_10_stocks_profit_arima:
        print(f"{ticker}: {profit:.2f} units of profit")
else:
    print("No stocks found with calculated cumulative profit for ARIMA.")


# --- Model 2: Exponential Smoothing (ETS) ---
# Simplified approach: Train on 80% of data, predict on 20% for each stock.
# If predicted return > 0, assume we take the actual daily return; otherwise, 0 return.
cumulative_profits_ets = {}
print("\nTraining ETS models and calculating profits...")

for ticker in tqdm(filtered_tickers, desc="Training ETS"):
    returns = daily_returns[ticker]
    
    # Ensure enough data for split and ETS fitting
    if len(returns) < 5:
        continue

    train_size = int(len(returns) * TRAIN_SPLIT_RATIO)
    train_data, test_data = returns[:train_size], returns[train_size:]

    if test_data.empty: # Check if test_data is empty after split
        continue

    try:
        # Fit Exponential Smoothing model (e.g., additive, no trend, no seasonality for returns)
        # 'initialization_method="estimated"' allows the model to estimate initial values.
        model = ExponentialSmoothing(train_data, trend=None, seasonal=None, initialization_method="estimated")
        model_fit = model.fit()
        
        # Forecast on the test set
        predictions = model_fit.predict(start=len(train_data), end=len(returns)-1)
        
        # Align predictions with test_data index
        predictions.index = test_data.index

        # Strategy: If predicted return is positive, assume we capture the actual return. Else, no position.
        strategy_returns = test_data.copy()
        strategy_returns[predictions <= 0] = 0 

        profit = calculate_cumulative_profit(strategy_returns)
        cumulative_profits_ets[ticker] = profit

    except Exception as e:
        # print(f"Could not train ETS for {ticker}: {e}") # Uncomment for detailed errors
        pass # Suppress frequent ETS errors for brevity in tqdm

# --- Print Top 10 Stocks by Cumulative Profit (ETS) ---
top_10_stocks_profit_ets = sorted(cumulative_profits_ets.items(), key=lambda item: item[1], reverse=True)[:TOP_N_STOCKS]

print(f"\n--- Top {TOP_N_STOCKS} Stocks by Cumulative Profit (Exponential Smoothing Model) ---")
if top_10_stocks_profit_ets:
    for ticker, profit in top_10_stocks_profit_ets:
        print(f"{ticker}: {profit:.2f} units of profit")
else:
    print("No stocks found with calculated cumulative profit for ETS.")




# --- Overall Comparison and Graphical Representation (Single Representative Stock) ---
print("\n--- Comparative Analysis of Models (Single Representative Stock) ---")

# Find the overall top stock from HMM for detailed comparison plots, if available
representative_ticker = None
if top_10_stocks_profit_hmm:
    representative_ticker = top_10_stocks_profit_hmm[0][0]
else: # Fallback if HMM had no top stocks
    # Try to pick a representative from other models if HMM is empty
    all_top_stocks = []
    if top_10_stocks_profit_arima: all_top_stocks.extend(top_10_stocks_profit_arima)
    if top_10_stocks_profit_ets: all_top_stocks.extend(top_10_stocks_profit_ets)
    if top_10_stocks_profit_sma: all_top_stocks.extend(top_10_stocks_profit_sma)
    
    if all_top_stocks:
        # Pick the best performing stock overall from any model as representative
        representative_ticker = sorted(all_top_stocks, key=lambda item: item[1], reverse=True)[0][0]


if representative_ticker and representative_ticker in daily_returns and representative_ticker in stock_data:
    print(f"\nPlotting cumulative performance comparison for a representative stock: {representative_ticker}")
    
    # Calculate cumulative values for the representative stock for each model
    cumulative_values_comparison = {}

    # --- HMM Cumulative Value ---
    if representative_ticker in stock_predicted_returns_hmm:
        cumulative_values_comparison["HMM"] = (1 + stock_predicted_returns_hmm[representative_ticker]).cumprod() * 100
    
    # --- ARIMA Cumulative Value ---
    returns_rep = daily_returns[representative_ticker]
    prices_rep = stock_data[representative_ticker]
    
    train_size_rep = int(len(returns_rep) * TRAIN_SPLIT_RATIO)
    train_data_rep, test_data_rep = returns_rep[:train_size_rep], returns_rep[train_size_rep:]
    
    if not test_data_rep.empty:
        try:
            model_arima_rep = ARIMA(train_data_rep, order=(1,0,1))
            model_arima_fit_rep = model_arima_rep.fit()
            predictions_arima_rep = model_arima_fit_rep.predict(start=len(train_data_rep), end=len(returns_rep)-1)
            predictions_arima_rep.index = test_data_rep.index
            strategy_returns_arima_rep = test_data_rep.copy()
            strategy_returns_arima_rep[predictions_arima_rep <= 0] = 0
            cumulative_values_comparison["ARIMA"] = (1 + strategy_returns_arima_rep).cumprod() * 100
        except Exception as e:
            print(f"Could not calculate ARIMA cumulative value for {representative_ticker}: {e}")

    # --- ETS Cumulative Value ---
    if not test_data_rep.empty:
        try:
            model_ets_rep = ExponentialSmoothing(train_data_rep, trend=None, seasonal=None, initialization_method="estimated")
            model_ets_fit_rep = model_ets_rep.fit()
            predictions_ets_rep = model_ets_fit_rep.predict(start=len(train_data_rep), end=len(returns_rep)-1)
            predictions_ets_rep.index = test_data_rep.index
            strategy_returns_ets_rep = test_data_rep.copy()
            strategy_returns_ets_rep[predictions_ets_rep <= 0] = 0
            cumulative_values_comparison["ETS"] = (1 + strategy_returns_ets_rep).cumprod() * 100
        except Exception as e:
            print(f"Could not calculate ETS cumulative value for {representative_ticker}: {e}")

    


    # Plotting Cumulative Values for Comparison
    if cumulative_values_comparison:
        plt.figure(figsize=(14, 7))
        for model_name, cum_values in cumulative_values_comparison.items():
            if not cum_values.empty:
                plt.plot(cum_values.index, cum_values, label=f'{model_name} Strategy')
        
        # Add Buy & Hold benchmark
        if representative_ticker in daily_returns and not daily_returns[representative_ticker].empty:
            buy_hold_returns = daily_returns[representative_ticker]
            cumulative_buy_hold = (1 + buy_hold_returns).cumprod() * 100
            plt.plot(cumulative_buy_hold.index, cumulative_buy_hold, label='Buy & Hold', linestyle='--', color='gray')

        plt.title(f'Cumulative Value Comparison for {representative_ticker} Across Different Models')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Value (Starting at 100)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"No cumulative value data available for comparison plots for {representative_ticker}.")

else:
    print("\nCould not find a representative stock to plot comparison due to data issues or insufficient data for all models.")


# --- Plotting Cumulative Performance for Top 10 HMM Stocks vs. Buy & Hold ---
print(f"\n--- Plotting Cumulative Performance for Top {TOP_N_STOCKS} HMM Stocks vs. Buy & Hold ---")

if not top_10_stocks_profit_hmm:
    print("No top 10 HMM stocks to plot.")
else:
    for ticker, _ in top_10_stocks_profit_hmm:
        if ticker in stock_predicted_returns_hmm and ticker in daily_returns:
            hmm_returns = stock_predicted_returns_hmm[ticker]
            buy_hold_returns = daily_returns[ticker]

            if not hmm_returns.empty and not buy_hold_returns.empty:
                cumulative_hmm_value = (1 + hmm_returns).cumprod() * 100
                cumulative_buy_hold_value = (1 + buy_hold_returns).cumprod() * 100

                plt.figure(figsize=(12, 6))
                plt.plot(cumulative_hmm_value.index, cumulative_hmm_value, label='HMM Strategy', color='blue')
                plt.plot(cumulative_buy_hold_value.index, cumulative_buy_hold_value, label='Buy & Hold', linestyle='--', color='red')
                
                plt.title(f'Cumulative Value for {ticker}: HMM Strategy vs. Buy & Hold')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Value (Starting at 100)')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Skipping plot for {ticker}: Insufficient data for HMM or Buy & Hold returns.")
        else:
            print(f"Skipping plot for {ticker}: Data not found in HMM predicted returns or daily returns.")


# --- Analysis of Convergence/Divergence (re-iterated for clarity) ---
print("\n--- Re-analysis of Convergence/Divergence ---")
print("1. Stock Prices: Generally diverge. They do not converge to a finite value over long periods, reflecting market growth or decline.")
print("2. Daily Returns: Tend to be stationary. Their statistical properties (mean, variance) typically converge to stable values over time.")
print("3. HMM Hidden States: The HMM model's parameters (transition and emission probabilities) converge during the training process (EM algorithm). The sequence of inferred states will continue to evolve with new data, reflecting changing market regimes.")
print("4. Cumulative Profit: For any strategy that consistently generates a positive average return, the cumulative profit will generally diverge (grow towards infinity) due to compounding. There is no finite upper limit to wealth accumulation in a continuously growing market.")
print("5. Limit of Highest Number of Days with Positive Daily Return: This is not a concept that has a mathematical 'limit' in the traditional sense. The *proportion* of positive days tends to converge to a probability (e.g., 50-60% for many markets). The *count* of consecutive positive days is a statistical observation that can fluctuate, and while it might increase with longer observation periods, it doesn't converge to a fixed number.")
