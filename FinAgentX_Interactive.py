# === 🧠 FinAgentX - Autonomous Trading Agent ===
# Combined Interactive Notebook for Google Colab

# === STEP 1: Install Dependencies ===
!pip install yfinance pandas numpy torch transformers langchain faiss-cpu ray -q
!pip install yfinance pandas numpy torch transformers langchain faiss-cpu ray[rllib] -q
!pip install -U langchain-community
!pip install -U ray[rllib] -q
# === STEP 2: Import Libraries ===
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
import gym
from gym import spaces
from ray.rllib.algorithms.ppo import PPO

# === STEP 3: Market Data Retrieval ===
def get_historical_data(ticker="AAPL", period="6mo", interval="1d"):
    print("📉 Downloading historical data...")
    data = yf.download(ticker, period=period, interval=interval)
    # Check if data is empty and raise an exception if so
    if data.empty:
        raise ValueError("Downloaded data is empty. Check ticker, period, and interval.")
    return data['Close'].values

# === STEP 4: Market Sentiment Analyzer (RAG Placeholder) ===
class MarketSentimentAnalyzer:
    def __init__(self):
        print("🔍 Initializing sentiment pipeline...")
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        result = self.sentiment_pipeline(text)[0]
        return result['label'], result['score']

# === STEP 5: Custom Trading Environment ===
# Import gymnasium instead of gym
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env): # Inherit from gymnasium.Env
    def __init__(self, prices):
        self.prices = prices
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0

        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Print environment spaces during initialization
        print("Action Space:", self.action_space)
        print("Observation Space:", self.observation_space)

        # Ensure prices has more than one element
        if len(self.prices) < 2:
            raise ValueError("Prices array must contain at least 2 elements.")

    def reset(self, seed=None, options=None): # Add seed and options for Gymnasium compatibility
        super().reset(seed=seed) # Call super().reset() in Gymnasium environments
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        # Return the initial observation
        return self._get_observation(), {} # Return observation and info dictionary

    def step(self, action):
        self.current_step += 1
        
        #Prevent stepping beyond the available price data
        if self.current_step >= len(self.prices):
            return self._get_observation(), 0, True, False, {} # Return observation, reward, terminated, truncated, info


        current_price = self.prices[self.current_step]  
        
        if action == 0:  # Buy
            # ... (buy logic)
            pass 
        elif action == 1:  # Sell
            # ... (sell logic)
            pass
        else:  # Hold
            pass  

        reward = self._calculate_reward(current_price)  
        # Use terminated and truncated for Gymnasium compatibility
        terminated = self.current_step == len(self.prices) - 1  
        truncated = False # Set to True if episode ends prematurely due to other reasons

        return self._get_observation(), reward, terminated, truncated, {} # Return all required values
    
    def _get_observation(self):
        # Assuming your observation is [balance, holdings]
        return np.array([self.balance, self.holdings], dtype=np.float32)
    
    def _calculate_reward(self, current_price):
        #Implement your reward calculation logic here
        return 0  # Replace with your reward function

# === STEP 6: Train RL Agent ===
def train_rl_agent(prices):
    print("🏋️ Training PPO agent...")
    
    # Define a function to create the environment, accepting a config dictionary
    def env_creator(config):
        # Access the prices data from the config dictionary
        prices_data = config.get("prices", prices)  # Use default prices if not in config
        return TradingEnv(prices_data)

    # Use the env_creator function to register the environment
    # Provide a unique environment ID (e.g., "TradingEnv-v0")
    from ray.tune.registry import register_env
    register_env("TradingEnv-v0", env_creator)

    # Use the registered environment ID when instantiating PPO
    trainer = PPO(env="TradingEnv-v0", config={
        "env_config": {"prices": prices},  # Pass prices data in the config
        "framework": "torch",
        # Reducing the number of workers to 1 may avoid the out-of-range error.
        "num_workers": 1,  
        # Explicitly set remote_worker_envs to False to disable remote envs
        "remote_worker_envs": False, 
        # Explicitly set create_env_on_driver to True to create the env on the driver
        "create_env_on_driver": True,
    })

    for i in range(5):
        result = trainer.train()
        print(result.keys())
        # Check if the key exists before accessing it
        if "episode_reward_mean" in result:
            print(f"✅ Iteration {i + 1} | Avg. Reward: {result['episode_reward_mean']:.2f}")
            reward = result.get("episode_reward_mean") or result.get("episode_return_mean")
            print(f"Iteration {i}: reward = {reward}")

        else:
            print(f"⚠️ Iteration {i + 1} | Episode reward mean not found in results.")
            # You can print the entire result dictionary for debugging:
            # print(f"Result dictionary: {result}")

    return trainer

# === STEP 7: Evaluate Agent (Accuracy Checker) ===
def evaluate_agent(trainer, prices):
    print("🧪 Evaluating trained agent...")
    env = TradingEnv(prices)
    obs = env.reset()
    total_reward = 0
    done = False

    # Get the policy directly from the trainer
    policy = trainer.get_policy()  # This line was changed

    while not done:
        # Use compute_single_action to get the action
        action = policy.compute_single_action(obs)  
        obs, reward, done, _, _ = env.step(action)  # Update to match Gymnasium's step output
        total_reward += reward

    final_value = env.balance + env.holdings * prices[env.current_step]
    print(f"📈 Final Portfolio Value: ${final_value:.2f}")
    print(f"🧮 Total Reward: {total_reward:.2f}")
    print("🧪 Evaluation complete.")

# === STEP 8: Execute All Steps ===
print("🚀 Starting FinAgentX pipeline...")

# 1. Get data
prices = get_historical_data("AAPL")

# 2. Analyze market sentiment
sentiment_analyzer = MarketSentimentAnalyzer()
news = "Apple stock is expected to outperform expectations this quarter."
label, score = sentiment_analyzer.analyze_sentiment(news)
print(f"📰 Sentiment: {label} (Confidence: {score:.2f})")

# 3. Train agent
trained_agent = train_rl_agent(prices)

# 4. Evaluate performance
evaluate_agent(trained_agent, prices)

print("✅ FinAgentX complete.")
