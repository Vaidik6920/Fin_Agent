import pytest
import numpy as np
from finagentx.accuracy_checker import AccuracyChecker

def test_accuracy_checker():
    prices = np.array([100, 101, 102, 103, 104])
    portfolio_values = [10000, 10100, 10200, 10300, 10400]  # Sample portfolio values from agent's actions
    
    accuracy_checker = AccuracyChecker(prices)
    performance = accuracy_checker.evaluate_agent(portfolio_values)
    
    # Test if accuracy is calculated properly
    assert isinstance(performance, dict)
    assert "accuracy_percentage" in performance
    assert performance["accuracy_percentage"] >= 0  # Accuracy should be >= 0
    assert performance["agent_return"] >= 0  # Agent return should be >= 0
    assert performance["buy_and_hold_return"] >= 0  # Buy-and-hold return should be >= 0

def test_accuracy_checker_no_profit():
    prices = np.array([100, 101, 102, 103, 104])
    portfolio_values = [10000, 10000, 10000, 10000, 10000]  # No profit scenario (buy and hold)
    
    accuracy_checker = AccuracyChecker(prices)
    performance = accuracy_checker.evaluate_agent(portfolio_values)
    
    # Test if no profit results in accuracy of 0%
    assert performance["accuracy_percentage"] == 0
    assert performance["agent_return"] == 0
    assert performance["buy_and_hold_return"] == 0
