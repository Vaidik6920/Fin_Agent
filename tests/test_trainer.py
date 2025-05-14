import pytest
from finagentx.agent.trainer import train_rl_agent
from finagentx.data.market_data import get_historical_data

def test_train_rl_agent():
    prices = get_historical_data("AAPL", period="1y", interval="1d")
    
    # Check if the agent can train without any errors
    try:
        trainer = train_rl_agent(prices)
        assert trainer is not None  # The trainer object should not be None
    except Exception as e:
        pytest.fail(f"Training failed: {e}")
