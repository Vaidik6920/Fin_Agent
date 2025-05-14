import yfinance as yf

def get_historical_data(ticker, period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    return data['Close'].values
