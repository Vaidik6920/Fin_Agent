import pytest
import yfinance as yf
from finagentx.data.market_data import get_historical_data

def test_get_historical_data():
    # Test if the market data is fetched correctly for a known ticker
    prices = get_historical_data("AAPL", period="1d", interval="1m")
    assert len(prices) > 0  # There should be some data returned
    assert isinstance(prices, np.ndarray)  # Data should be a numpy array
    
    # Test for different period and interval
    prices_1y = get_historical_data("AAPL", period="1y", interval="1d")
    assert len(prices_1y) > 250  # A full year of daily data should have > 250 data points

def test_get_historical_data_invalid_ticker():
    # Test for invalid ticker symbol
    try:
        prices = get_historical_data("INVALID_TICKER")
        assert len(prices) == 0  # Expect empty data
    except Exception as e:
        assert isinstance(e, ValueError)  # Expect a ValueError to be raised for invalid tickers
