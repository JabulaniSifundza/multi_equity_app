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
import fredapi
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec

api_key = st.secrets["GEMINI_API_KEY"]
fred_api_key = st.secrets["FRED_API_KEY"]

fred = fredapi.Fred(api_key=fred_api_key)

def get_treasury_yield(series_id):
    """
    Fetches Treasury yield data from FRED.

    Args:
        series_id (str): The FRED series ID for the desired Treasury yield.

    Returns:
        pandas.Series: The Treasury yield time series, or None if an error occurs.
    """
    try:
        return fred.get_series(series_id)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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
    research_ticker = st.text_input("Enter the ticker symbol 👇🏾", placeholder="Ticker symbol", key="capmInput")
    if len(research_ticker) < 1:
        st.write("Please enter a ticker symbol to start")
    else:
        company_ticker = Ticker(research_ticker)
        market_ticker = Ticker('^GSPC')
        stock_priceDF = company_ticker.history(period='1d', start='2018-01-31', end='2023-02-01')
        market_DF = market_ticker.history(period='1d', start='2018-01-31', end='2023-02-01')
        stock_priceDF['log_returns'] = np.log(stock_priceDF['adjclose']/stock_priceDF['adjclose'].shift(1))
        market_DF['log_returns'] = np.log(market_DF['adjclose']/market_DF['adjclose'].shift(1))
        stock_priceDF = stock_priceDF.dropna()
        market_DF = market_DF.dropna()
        combined = pd.concat([stock_priceDF, market_DF], axis=0, join='inner')
        covariance = (np.cov(stock_priceDF['log_returns'], market_DF['log_returns'])) * 250
        covariance_with_market = covariance[0, 1]
        market_variance = market_DF['log_returns'].var() * 250
        beta_final = covariance_with_market / market_variance
        # Calating the company's CAPM/Expected Return
        ten_year_yield = get_treasury_yield('GS10')
        print(type(ten_year_yield))
        # print((np.log(market_DF['adjclose']/market_DF['adjclose'].shift(1)).mean() * 252)*100)
        capm_expected_return = ten_year_yield.iloc[-1] + beta_final * (((np.log(market_DF['adjclose']/market_DF['adjclose'].shift(1)).mean() * 252)*100) - ten_year_yield.iloc[-1])
        #print(f"CAPM Expected Return: {capm_expected_return:.2f}%")
        #company_capm = 0.025 + beta_final * 0.05
        st.write(f"The CAPM Expected Return for {research_ticker} is {capm_expected_return:.2f}%")
        income_state = company_ticker.income_statement()
        # Getting the company's Balance sheet
        balance_sheet = company_ticker.balance_sheet()
        cash_flow_statement = company_ticker.cash_flow(trailing=False)
        # Functions to clean up cost, profit, liquidity metrcis and separate these values into their own objects
        def get_costs_(cost_array):
            return [float(cost) for cost in cost_array]

        def get_year(income_state_years):
            years = []
            for year in income_state_years:
                str_time = year.strftime('%Y-%m-%d')
                years.append(str_time)
            return years

        def get_net(ebit_array):
            return [float(earning) for earning in ebit_array]
            
        def get_ebit(earnings_arr):
            return [float(earned) for earned in earnings_arr]


        def get_revenue(revenue_arr):
            return [float(rev) for rev in revenue_arr]
        years = get_year(income_state['asOfDate'])
        total_expense = get_costs_(income_state['TotalExpenses'])
        net = get_net(income_state['NetIncome'])
        ebit = get_ebit(income_state['EBIT'])
        total_revenues = get_revenue(income_state['TotalRevenue'])
        total_expense_dict = {year: cost for (year, cost) in zip(years, total_expense) if cost > 0}
        net_income_dict = {year: income for (year, income) in zip(years, net) if income > 0}
        for year,income in net_income_dict.items():
            if income > 0:
                st.write(f"Net Income for {year} was ${income:,.2f}")
                
        for year,cost in total_expense_dict.items():
            if cost > 0:
                st.write(f"Total Expenses for {year} was ${cost:,.2f}")
                
        def get_total_liabilities(liabilities):
            return [float(liability) for liability in liabilities]
            

        def get_total_assets(assets):
            return [float(asset) for asset in assets]
            

        def get_total_cash(cash):
            return [float(liquidity) for liquidity in cash]


        def get_current_assets(current_assets):
            return [float(current_asset) for current_asset in current_assets]

        def get_current_liabilities(current_liabilities):
            return [float(current_liability) for current_liability in current_liabilities]


        def ending_cash_balance(ending_cash):
            return [float(ending_balance) for ending_balance in ending_cash]

        def get_operating_cash_flow(cash_flow_arr):
            return [float(cash_flow) for cash_flow in cash_flow_arr]

        ending_cash_balance = ending_cash_balance(cash_flow_statement['EndCashPosition'])
        total_liabilities = get_total_liabilities(balance_sheet['TotalLiabilitiesNetMinorityInterest'])
        total_assets = get_total_assets(balance_sheet['TotalAssets'])
        total_cash_equivalents = get_total_cash(balance_sheet['CashAndCashEquivalents'])
        total_current_assets = get_current_assets(balance_sheet['CurrentAssets'])
        total_get_current_liabilities = get_current_liabilities(balance_sheet['CurrentLiabilities'])
        operating_cash_flows = get_operating_cash_flow(cash_flow_statement['OperatingCashFlow'])
        # Calculating Ratios
        # Current ratio - company's ability to pay off its current liabilities 
        current_ratios = {year: current_asset/current_liability for(year, current_asset, current_liability) in zip(years, total_current_assets, total_get_current_liabilities)}
        for year, ratio in current_ratios.items():
            if not math.isnan(ratio):
                st.write(f"Current Ratio for {year} was {ratio:.2f}")
        # ROCE - Return on Capital Employed
        roce = {year: year_ebit/(assets - curr_liabilities) for(year, year_ebit, assets, curr_liabilities) in zip(years, ebit, total_assets, total_get_current_liabilities)}
        # Net Profit Margin
        net_profit_margin = {year: (net_income/revenue)*100 for(year, net_income, revenue) in zip(years, net, total_revenues)}
        # Operating Cash Flow ratio
        operating_cash_flow_ratio = {year: operating_cash/current_liability for(year, operating_cash, current_liability) in zip(years, operating_cash_flows, total_get_current_liabilities)}
        try:
            json_spec = JsonSpec(dict_=net_income_dict, max_value_length=4000)
            json_toolkit = JsonToolkit(spec=json_spec)
            json_agent_executor = create_json_agent(
                llm=model, toolkit=json_toolkit, verbose=True
            )
            results = json_agent_executor.run(
                "What are your thoughts on the profit trend? Use the values of the keys in the dictionary to answer the question. The keys are dates and the values are the net profits. Ignore the nan values and give an opionion of the trend of the data you have available. You are a highly analytical financial analyst looking for companies to invest in. Investments can be long or short positions."
            )
            # print(results)
            st.write(results)
        except Exception as Err:
            st.write("The HuggingFace Financial Summarization Model failed to load")
            print(Err)

