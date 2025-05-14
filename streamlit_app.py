import streamlit as st
import pandas as pd
import numpy as np
from finagentx.data.market_data import get_historical_data
from finagentx.nlp.sentiment_rag import MarketSentimentAnalyzer
from finagentx.agent.trading_env import TradingEnv
from finagentx.nlp.rag_pipeline import FinancialRAG
from finagentx.accuracy_checker import AccuracyChecker

# Load data
st.title("📈 FinAgentX - AI Trading Dashboard")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
prices = get_historical_data(ticker)
df = pd.DataFrame(prices, columns=["Close"])

st.line_chart(df["Close"], use_container_width=True)

# Run sentiment analysis
st.subheader("💬 Market Sentiment Analyzer")
user_input = st.text_area("Input news/headline:")
if user_input:
    analyzer = MarketSentimentAnalyzer()
    label, score = analyzer.analyze_sentiment(user_input)
    st.success(f"Sentiment: **{label}** (Confidence: {score:.2f})")

# Simulate environment step-by-step
st.subheader("🤖 Simulated Trading")
env = TradingEnv({"prices": prices})
obs = env.reset()
portfolio_values = []

for _ in range(len(prices) - 1):
    action = np.random.choice([0, 1, 2])  # Random policy (for now)
    obs, reward, done, _ = env.step(action)
    portfolio_value = env.balance + env.holdings * prices[env.current_step]
    portfolio_values.append(portfolio_value)
    if done:
        break

# Evaluate agent's performance
accuracy_checker = AccuracyChecker(prices)
performance = accuracy_checker.evaluate_agent(portfolio_values)

# Display results
st.line_chart(portfolio_values, use_container_width=True)
st.write(f"📊 Final Portfolio Value: ${portfolio_values[-1]:.2f}")

st.subheader("📉 Performance Evaluation")
st.write(f"Agent Final Portfolio Value: ${performance['agent_final_value']:.2f}")
st.write(f"Buy-and-Hold Final Portfolio Value: ${performance['buy_and_hold_value']:.2f}")
st.write(f"Agent Return: {performance['agent_return'] * 100:.2f}%")
st.write(f"Buy-and-Hold Return: {performance['buy_and_hold_return'] * 100:.2f}%")
st.write(f"**Accuracy**: {performance['accuracy_percentage']:.2f}% better than Buy-and-Hold strategy")
