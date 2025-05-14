import pytest
import numpy as np
from finagentx.agent.trading_env import TradingEnv

def test_trading_env_initialization():
    # Test if the environment initializes correctly
    prices = np.array([100, 101, 102, 103, 104])  # Sample price data
    env = TradingEnv(config={"prices": prices})
    
    # Test if initial balance and holdings are correctly set
    assert env.balance == 10000
    assert env.holdings == 0
    assert env.current_step == 0

def test_reset():
    # Test if the reset function works properly
    prices = np.array([100, 101, 102, 103, 104])
    env = TradingEnv(config={"prices": prices})
    
    env.reset()
    assert env.current_step == 0
    assert env.balance == 10000
    assert env.holdings == 0

def test_step():
    # Test if the step function works for different actions (Buy, Sell, Hold)
    prices = np.array([100, 101, 102, 103, 104])
    env = TradingEnv(config={"prices": prices})
    
    env.reset()
    
    # Test Buy action (action == 0)
    obs, reward, done, _ = env.step(0)
    assert env.balance < 10000  # Balance should decrease when buying
    assert env.holdings > 0  # Holdings should increase
    
    # Test Sell action (action == 1)
    obs, reward, done, _ = env.step(1)
    assert env.balance > 10000  # Balance should increase when selling
    assert env.holdings == 0  # Holdings should reset to 0
    
    # Test Hold action (action == 2)
    obs, reward, done, _ = env.step(2)
    assert env.balance == 10000  # Balance should stay the same when holding
    assert env.holdings == 0  # Holdings should stay the same when holding
