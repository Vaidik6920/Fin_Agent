import numpy as np

class AccuracyChecker:
    def __init__(self, prices):
        self.prices = prices

    def evaluate_agent(self, agent_portfolio_values):
        """
        Evaluate the performance of the agent's trading strategy.
        
        agent_portfolio_values: List of portfolio values from the agent's actions.
        """
        # Calculate the return of the agent's strategy
        agent_final_value = agent_portfolio_values[-1]
        agent_return = (agent_final_value - agent_portfolio_values[0]) / agent_portfolio_values[0]

        # Calculate the return of a Buy-and-Hold strategy (just buying and holding the stock)
        buy_and_hold_value = self.prices[-1] / self.prices[0] * 10000  # assuming initial balance is 10,000
        buy_and_hold_return = (buy_and_hold_value - 10000) / 10000

        # Accuracy: How much better is the agent compared to Buy-and-Hold?
        accuracy = (agent_return - buy_and_hold_return) * 100  # Return in percentage

        return {
            "agent_final_value": agent_final_value,
            "buy_and_hold_value": buy_and_hold_value,
            "accuracy_percentage": accuracy,
            "agent_return": agent_return,
            "buy_and_hold_return": buy_and_hold_return
        }
