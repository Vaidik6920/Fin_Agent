import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, config):
        self.prices = config["prices"]
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0

        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        return self._get_obs()

    def _get_obs(self):
        price = self.prices[self.current_step]
        return np.array([price, self.holdings], dtype=np.float32)

    def step(self, action):
        price = self.prices[self.current_step]
        reward = 0

        if action == 0:  # Buy
            num_shares = self.balance // price
            self.holdings += num_shares
            self.balance -= num_shares * price
        elif action == 1:  # Sell
            self.balance += self.holdings * price
            self.holdings = 0

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        portfolio_value = self.balance + self.holdings * price
        reward = portfolio_value - self.initial_balance

        return self._get_obs(), reward, done, {}
