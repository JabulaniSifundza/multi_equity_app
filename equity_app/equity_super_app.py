from bs4 import BeautifulSoup
import re
from datetime import datetime
import io
import math
import streamlit as st
import sys
from google import genai
import datetime
import numpy as np
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.prompts import PromptTemplate
from yahooquery import Ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

api_key = st.secrets["GEMINI_API_KEY"]

monte_carlo_simulation, portfolio_optimization, news_and_sentiment_analysis, financials_and_trend_analysis = st.tabs(["Monte Carlo Simulation", "Portfolio Optimization", "News and Sentiment Analysis", "Financials and Trend Analysis"])
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
with monte_carlo_simulation:
    st.write("Streamlit version")
    st.write(st.__version__)                
    st.subheader("Run Monte Carlo Simulations on a Selected Stock Price")
    monte_carlo_symbol = st.text_input("Enter the ticker symbol 👇🏾", placeholder="Ticker symbol", key="monteCarloInput")
    if len(monte_carlo_symbol) < 1:
        st.write("Please enter a ticker symbol to start")
    else:
        st.subheader(f"Running Monte Carlo Sims on {monte_carlo_symbol} stock price information")
        stock = monte_carlo_symbol
        monte_Df = Ticker(stock).history(period='1d', start='2015-03-12')
        # print(monte_Df)
        monte_Df = monte_Df.reset_index(level=None, drop=False)
        try:
            monte_Df['date'] = pd.to_datetime(monte_Df['date'], utc=True) #convert to UTC if possible
        except ValueError:
            monte_Df['date'] = pd.to_datetime(monte_Df['date']).dt.tz_localize(None)
        monte_Df['date'] = monte_Df['date'].dt.date #extract just the date.
        # print(monte_Df)
        days = (monte_Df['date'].iloc[-1] - monte_Df['date'].iloc[0]).days
        growth = monte_Df['adjclose'][2518] / monte_Df['adjclose'][0]
        num_years = days/365
        cagr = growth ** (1/num_years) - 1
        std_dev = monte_Df['adjclose'].pct_change().std()
        trading_days = 252
        std_dev = std_dev * math.sqrt(trading_days)
        print(cagr, std_dev)
        daily_return_pct = np.random.normal(cagr/trading_days,std_dev/math.sqrt(trading_days), trading_days)+1
        price_series = [monte_Df['adjclose'][2518]]
        price_series.extend(price_series[-1] * j for j in daily_return_pct)
        st.write("Daily Returns")
        st.line_chart(price_series)
        # Random walks
        num_trials = 3000
        closing_prices = []
        for _ in range(num_trials):
            daily_return_pct = np.random.normal(cagr/trading_days,std_dev/math.sqrt(trading_days), trading_days)+1
            price_series = [monte_Df['adjclose'][2518]]

            price_series.extend(price_series[-1] * j for j in daily_return_pct)
            closing_prices.append(price_series[-1])
            plt.plot(price_series)
        st.pyplot(plt)
            
        mean_end_price = round(np.mean(closing_prices), 2)
        st.write(f"Expected price (mean): $ {mean_end_price:.2f}")
        # To guage risk vs reward; split distributions into percentiles
        # top 10 possible outcomes
        top_ten = np.percentile(closing_prices, 100-10)
        # bottom 10 possible outcomes
        bottom_ten = np.percentile(closing_prices, 10)
        st.write(f"The last closing price was: $ {monte_Df['adjclose'].iloc[-1]:.2f}")
        st.write(f"Top 10th percentile expected simulated price $ {top_ten:.2f}")
        st.write(f"Bottom 10th percentile expected simulated price $ {bottom_ten:.2f}")
    
    
# Portfolo Optimization
with portfolio_optimization:
    st.subheader("Run Monte Carlo Simulations For Portfolio Optimization")
    portfolio_symbol = st.text_input("Enter the ticker symbols separated by commas👇🏾", placeholder="Ticker symbol", key="portfolioInput")
    if len(portfolio_symbol) < 1:
        st.write("Please enter portfolio Ticker symbols separated by commas")
    else:
        stock_symbols = portfolio_symbol.split(",")
        # print(stock_symbols)
        st.write("Your current Portfolio consists of: ", stock_symbols)
        data = Ticker(stock_symbols).history(start='2013-01-01', end='2023-01-01')
        print(data)
        port_asset_count = len(stock_symbols)
        asset_weights = np.random.random(port_asset_count)
        asset_weights /= np.sum(asset_weights)
        adj_close = pd.DataFrame(data['adjclose'])
        # Pivot the DataFrame so that Ticker symbols become column names
        df = adj_close.pivot_table(index='date', columns='symbol', values='adjclose')
        pf_log_returns = np.log(df/df.shift(1))
        avg_pf_returns = pf_log_returns.mean() * 252
        covariance_pf_returns = pf_log_returns.cov() * 252
        correlation_pf_returns = pf_log_returns.corr() * 252
        print(asset_weights.shape)
        print(pf_log_returns.shape)
        expected_pf_return = np.sum(asset_weights * pf_log_returns.mean()) * 252
        expected_pf_variance = np.dot(asset_weights.T, np.dot(pf_log_returns.cov() * 252, asset_weights))
        expected_pf_volatility = np.sqrt(np.dot(asset_weights.T, np.dot(pf_log_returns.cov() * 252, asset_weights)))
        pf_returns = []
        pf_volatilities = []
        for _ in range(1000):
            asset_weights = np.random.random(port_asset_count)
            asset_weights /= np.sum(asset_weights)
            pf_returns.append(np.sum(asset_weights * pf_log_returns.mean()) * 252)
            pf_volatilities.append(np.sqrt(np.dot(asset_weights.T, np.dot(pf_log_returns.cov() * 250, asset_weights))))
        pf_returns = np.array(pf_returns)
        pf_volatilities = np.array(pf_volatilities)
        sharpe_rat = expected_pf_return/expected_pf_volatility
        portfolios = pd.DataFrame({'Return': pf_returns, 'Volatility': pf_volatilities})
        num_ports = 125000
        rand_weights = np.zeros((num_ports, len(stock_symbols)))
        return_arr = np.zeros(num_ports)
        volatility_arr = np.zeros(num_ports)
        sharpe_ratio_arr = np.zeros(num_ports)
        port_length = len(stock_symbols)

        for iteration in range(num_ports):
            new_weights = np.array(np.random.random(port_length))
            new_weights = np.array(new_weights / np.sum(new_weights))
            rand_weights[iteration,:] = new_weights
            return_arr[iteration] = np.sum(new_weights * pf_log_returns.mean()) * 252
            volatility_arr[iteration] = np.sqrt(np.dot(new_weights.T, np.dot(pf_log_returns.cov() * 252, new_weights)))
            sharpe_ratio_arr[iteration] = return_arr[iteration]/volatility_arr[iteration]
        max_sharpe = sharpe_ratio_arr.max()
        max_cord_val = sharpe_ratio_arr.argmax() 
        max_sr_ret = return_arr[max_cord_val]
        max_sr_vol = volatility_arr[max_cord_val]
        ideal_weights_arr = rand_weights[max_cord_val,:]
        monte_ports = pd.DataFrame({'Return': return_arr, 'Volatility': volatility_arr, 'Sharpe_Ratio':sharpe_ratio_arr})
        monte_ports.plot(x='Volatility', y='Return', kind='scatter', c='Sharpe_Ratio', edgecolors='black',cmap='RdYlGn',figsize=(10, 6))
        plt.xlabel('Expected Portfolio Volatility')
        plt.ylabel('Expected Portfolio Return')
        st.pyplot(plt)
        st.subheader("From 125,000 random walks of portfolio weight allocations")
        st.write(f"Maximum sharpe ratio: {max_sharpe:.2f}")
        # print(max_cord_val)
        st.write(f"Efficient portfolio expected return: {100 * max_sr_ret:.2f}%")
        st.write(f"Efficient portfolio portfolio volatility: {max_sr_vol:.2f}")
        for idx in range(len(stock_symbols)):
            st.write(f"Ideal portfolio security weight => {stock_symbols[idx]}: {100 * ideal_weights_arr[idx]:.2f}%")
        # print(ideal_weights_arr)
        
        
# News and Sentiment Analysis
with news_and_sentiment_analysis:
    st.subheader("Individual News and Sentiment Analysis")
    st.write("Please Enter a ticker symbol to begin research")
    research_ticker = st.text_input("Enter the ticker symbol 👇🏾", placeholder="Ticker symbol", key="sentimentInput")
    if len(research_ticker) < 1:
        st.write("Please enter a ticker symbol to start")
    else:
        st.subheader(f"Retrieving {research_ticker} stock information. Uses Langchain and the Google Gemini API to retrieve news and sentiment analysis")
        try:
            system_message = "You are a highly analytical financial analyst looking for companies to invest in. Investments can be long or short positions. Analyze the sentiment of the text and how it might affect stock prices in the near future.."
            # This could also be a SystemMessage object
            # system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")
            langgraph_agent_executor = create_react_agent(model, tools=[YahooFinanceNewsTool()], prompt=system_message)
            query = f"What's your outlook on {research_ticker}"
            messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
            st.write(messages["messages"][-1].content)
        except Exception as Err:
            st.write("The HuggingFace Financial Summarization Model failed to load")
            print(Err)

with financials_and_trend_analysis:
    st.subheader("Financials and Trend Analysis")
    st.write("This feature is currently under construction")
    