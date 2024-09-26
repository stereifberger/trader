import pandas as pd
from alpaca_trade_api.rest import REST
import datetime

api = REST('PKY0AF1LLQM4SHJT290O', 'GMP58HBD35EhHhNBJAK46tsEUp2S6Ab0o3LGE7mc', 'https://paper-api.alpaca.markets')

def get_stock_data(symbol, start_date, end_date):
    barset = api.get_bars(symbol, TimeFrame.Minute, start=start_date, end=end_date)
    df = barset[symbol].df
    return df

def categorize_stocks(stock_list, liquidity_threshold):
    liquid = []
    illiquid = []
    for stock in stock_list:
        avg_volume = api.get_barset(stock, 'day', limit=1)[stock][0].v
        if avg_volume >= liquidity_threshold:
            liquid.append(stock)
        else:
            illiquid.append(stock)
    return liquid, illiquid

def generate_stock_dataframes(stock_list, start_date, end_date, liquidity_threshold):
    liquid_stocks, illiquid_stocks = categorize_stocks(stock_list, liquidity_threshold)
    stock_dataframes = {}
    
    for stock in liquid_stocks + illiquid_stocks:
        df = get_stock_data(stock, start_date, end_date)
        stock_dataframes[stock] = df
    
    return stock_dataframes, liquid_stocks, illiquid_stocks

def prepare_train_test_data(stock_dataframes, X, Y, Z):
    """
    Prepare training and testing data for stock price prediction.
    - X: number of minutes for the input window
    - Y: prediction horizon (minutes)
    - Z: threshold for price increase (%)
    """
    data = []
    for stock, df in stock_dataframes.items():
        df['future_price'] = df['close'].shift(-Y)
        df['price_increase'] = (df['future_price'] - df['close']) / df['close'] * 100
        df['target'] = (df['price_increase'] > Z).astype(int)
        df.dropna(inplace=True)
        
        for i in range(len(df) - X):
            features = df.iloc[i:i+X][['open', 'high', 'low', 'close', 'volume']].values
            target = df.iloc[i+X]['target']
            data.append((features, target))
    
    return data
