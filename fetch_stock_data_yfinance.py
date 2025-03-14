# fetch_stock_data_yfinance.py
import yfinance as yf
import pandas as pd
import sys



def fetch_stock_data(ticker, start_date, end_date):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data  # Return the stock data                                                                                                                                                              
                                                                            
def main():
    # Get the command line arguments
    ticker = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print(stock_data)
    stock_data.to_csv('stock_data.csv')
    print('Stock data saved to stock_data.csv')                      




if __name__ == '__main__':
    main() # Call the main function

# Run the script from the command line
# python fetch_stock_data_yfinance.py AAPL 2020-01-01 2020-12-31    
# python fetch_stock_data_yfinance.py AAPL 2020-01-01 2020-12-31
# python fetch_stock_data_yfinance.py AAPL 2020-01-01 2020-12-31
# python fetch_stock_data_yfinance.py AAPL 2020-01-01 2020-12-31
