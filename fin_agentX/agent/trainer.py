from ray.rllib.agents.ppo import PPOTrainer
from finagentx.agent.trading_env import TradingEnv

def train_rl_agent(prices):
    trainer = PPOTrainer(env=TradingEnv, config={
        "env_config": {"prices": prices},
        "framework": "torch",
        "num_workers": 1
    })
    for i in range(10):
        result = trainer.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
    return trainer
