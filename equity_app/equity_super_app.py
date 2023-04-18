from yahooquery import Ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from bs4 import BeautifulSoup
import re
from datetime import datetime
import io
import math
import streamlit as st

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/human-centered-summarization/financial-summarization-pegasus"
SENTI_MODEL_URL = "https://api-inference.huggingface.co/models/ahmedrachid/FinancialBERT-Sentiment-Analysis"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
sentiment = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

unwanted_string_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

st.title("Equity Research & Portfolio Optimization Helper App")                
st.subheader("Run Monte Carlo Simulations on a Selected Stock Price")
monte_carlo_symbol = st.text_input("Enter the ticker symbol 👇🏾", placeholder="Ticker symbol", key="monteCarloInput")
if len(monte_carlo_symbol) < 1:
    st.write("Please enter a ticker symbol to start")
else:
    st.subheader(f"Running Monte Carlo Sims on {monte_carlo_symbol} stock price information")
    stock = monte_carlo_symbol
    monte_Df = Ticker(stock).history(period='1d', start='2013-04-12')
    monte_Df.reset_index(level=None, drop=False, inplace=True)
    monte_Df['date'] = pd.to_datetime(monte_Df['date']).dt.date
    days = (monte_Df['date'].iloc[-1] - monte_Df['date'].iloc[0]).days
    growth = monte_Df['adjclose'][2518] / monte_Df['adjclose'][0]
    num_years = days/365
    cagr = growth ** (1/num_years) - 1
    std_dev = monte_Df['adjclose'].pct_change().std()
    trading_days = 252
    std_dev = std_dev * math.sqrt(trading_days)

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
    st.write(f"Expected price: $ {mean_end_price:.2f}")
    # To guage risk vs reward; split distributions into percentiles
    # top 10 possible outcomes
    top_ten = np.percentile(closing_prices, 100-10)
    # bottom 10 possible outcomes
    bottom_ten = np.percentile(closing_prices, 10)
    st.write(f"Top 10th percentile expected simulated price $ {top_ten:.2f}")
    st.write(f"Bottom 10th percentile expected simulated price $ {bottom_ten:.2f}")
    

st.subheader("Run Monte Carlo Simulations For Portfolio Optimization")
portfolio_symbol = st.text_input("Enter the ticker symbols separated by commas👇🏾", placeholder="Ticker symbol", key="portfolioInput")
if len(portfolio_symbol) < 1:
    st.write("Please enter portfolio Ticker symbols separated by commas")
else:
    stock_symbols = portfolio_symbol.split(",")
    st.write("Your current Portfolio consists of: ", stock_symbols)
    data = Ticker(stock_symbols).history(start='2013-01-01', end='2023-01-01')
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
    
st.subheader("Individual News and Sentiment Analysis")
st.write("Please Enter a ticker symbol to begin research")
research_ticker = st.text_input("Enter the ticker symbol 👇🏾", placeholder="Ticker symbol", key="sentimentInput")
research_tickers = []
if len(research_ticker) < 1:
    st.write("Please enter a ticker symbol to start")
else:
    st.subheader(f"Retrieving {research_ticker} stock information")
    st.write("Please be aware that the HuggingFace Financial Summarization Model takes a while to load and it often fails to load before the server tines out. Please be patient and attempt your search again if you get an error.")
    try:
        research_tickers.append(research_ticker)
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        def sent_query(payload):
            response = requests.post(SENTI_MODEL_URL, headers=headers, json=payload)
            return response.json()
        def get_news(ticker):
            news_source = f"https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws"
            r = requests.get(news_source)
            soup = BeautifulSoup(r.text, 'html.parser')
            linktags = soup.find_all('a')
            return [link['href'] for link in linktags]
        article_links = {ticker:get_news(ticker) for ticker in research_tickers}
        # print(article_links)
        def remove_unwanted_strings(urls, unwanted_string):
            # sourcery skip: invert-any-all
            new_urls = []
            for url in urls:
                if 'https://' in url and not any(exclude_word in url for exclude_word in unwanted_string):
                    res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                    new_urls.append(res)
            return list(set(new_urls))
        cleaned_urls = {ticker:remove_unwanted_strings(article_links[ticker], unwanted_string_list) for ticker in research_tickers}
      
        # print(cleaned_urls)
        company_ticker = Ticker(research_ticker)
        market_ticker = Ticker('^GSPC')
        stock_priceDF = company_ticker.history(period='1d', start='2018-01-31', end='2023-02-01')
        market_DF = market_ticker.history(period='1d', start='2018-01-31', end='2023-02-01')
        stock_priceDF['log_returns'] = np.log(stock_priceDF['adjclose']/stock_priceDF['adjclose'].shift(1))
        market_DF['log_returns'] = np.log(market_DF['adjclose']/market_DF['adjclose'].shift(1))
        stock_priceDF = stock_priceDF.dropna()
        market_DF = market_DF.dropna()
        covariance = (np.cov(stock_priceDF['log_returns'], market_DF['log_returns'])) * 250
        covariance_with_market = covariance[0, 1]

        market_variance = market_DF['log_returns'].var() * 250

        beta_final = covariance_with_market / market_variance
        company_capm = 0.025 + beta_final * 0.05
        
        st.write(f"Company Beta: {beta_final:.2f}")
        st.write(f"CAPM/Expected Return: {100*company_capm:.2f}%")
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
            return [float(ending_cash) for ending_cash in ending_cash]
        def get_operating_cash_flow(cash_flow_arr):
            return [float(cash_flow) for cash_flow in cash_flow_arr]
        
        income_state = company_ticker.income_statement()
        balance_sheet = company_ticker.balance_sheet()
        cash_flow_statement = company_ticker.cash_flow(trailing=False)
        
        net = get_net(income_state['NetIncome'])
        years = get_year(income_state['asOfDate'])
        total_expense = get_costs_(income_state['TotalExpenses'])
        ebit = get_ebit(income_state['EBIT'])
        total_revenues = get_revenue(income_state['TotalRevenue'])
        total_expense_dict = {year: cost for (year, cost) in zip(years, total_expense)}
        st.subheader("Total Expenses (Last 5 years)")
        st.line_chart(total_expense_dict)
        st.write("Expense Values")
        for exp in total_expense:
            st.write(f"$ {exp:,.2f}")
        st.subheader("Net Profit Totals (Last 5 years)")
        net_income_dict = {year: income for (year, income) in zip(years, net)}
        st.line_chart(net_income_dict)
        st.write("Net Profit Values")
        for profit in net:
            st.write(f"$ {profit:,.2f}")
    
        val_ending_cash_balance = ending_cash_balance(cash_flow_statement['EndCashPosition'])
        total_liabilities = get_total_liabilities(balance_sheet['TotalLiabilitiesNetMinorityInterest'])
        total_assets = get_total_assets(balance_sheet['TotalAssets'])
        total_cash_equivalents = get_total_cash(balance_sheet['CashAndCashEquivalents'])
        total_current_assets = get_current_assets(balance_sheet['CurrentAssets'])
        total_get_current_liabilities = get_current_liabilities(balance_sheet['CurrentLiabilities'])
        operating_cash_flows = get_operating_cash_flow(cash_flow_statement['OperatingCashFlow'])
        print(net)
        
        print(val_ending_cash_balance)
        print(total_expense)
        # Calculating Ratios
        # Current ratio - company's ability to pay off its current liabilities 
        current_ratios = {year: current_asset/current_liability for(year, current_asset, current_liability) in zip(years, total_current_assets, total_get_current_liabilities)}
        # ROCE - Return on Capital Employed
        roce = {year: year_ebit/(assets - curr_liabilities) for(year, year_ebit, assets, curr_liabilities) in zip(years, ebit, total_assets, total_get_current_liabilities)}
        print(roce)
        # Net Profit Margin
        
        net_profit_margin = {year: (net_income/revenue) for(year, net_income, revenue) in zip(years, net, total_revenues)}
        print(net_profit_margin)
        # Operating Cash Flow ratio
        operating_cash_flow_ratio = {year: operating_cash/current_liability for(year, operating_cash, current_liability) in zip(years, operating_cash_flows, total_get_current_liabilities)}
        
        def scrape_and_read_articles(URLs):
            NEWS_ARTICLES = []
            for url in URLs:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                paragraph_text = [paragraph.text for paragraph in paragraphs]
                words = ' '.join(paragraph_text).split(' ')[:350]
                full_article = ' '.join(words)
                NEWS_ARTICLES.append(full_article)
            return NEWS_ARTICLES
        articles = {ticker:scrape_and_read_articles(cleaned_urls[ticker]) for ticker in research_tickers}
        print(articles)
        
        def summarize(articles):
            summaries = []
            for article in articles:
                summary = query({"inputs": article})
                summaries.append(summary)
            return summaries
        ticker_summary = {ticker:summarize(articles[ticker]) for ticker in research_tickers}
        summary_text = [summarize(articles[ticker]) for ticker in research_tickers]
        print(summary_text)
        def get_sentiment(summaries):
            sentiments = []
            for summary in summaries:
                score = sentiment(summary[0]['summary_text'])
                score = sent_query(summary[0]['summary_text'])
                sentiments.append(score[0][0])
                # sentiments.append(score[0])
            return sentiments
        ticker_score = {ticker:get_sentiment(ticker_summary[ticker]) for ticker in research_tickers}
        def create_output_list(ticker_summary, scores, article_urls):
            output = []
            for ticker in research_tickers:
                for counter in range(len(ticker_summary[ticker])):
                    desired_output = [
                        ticker,
                        ticker_summary[ticker][counter][0]['summary_text'],
                        scores[ticker][counter]['label'],
                        scores[ticker][counter]['score'],
                        article_urls[ticker][counter]
                    ]
                    output.append(desired_output)
            return output
        full_summary = create_output_list(ticker_summary, ticker_score, cleaned_urls)
        full_summary.insert(0, ['Ticker Symbol', 'Article Summary', 'Sentiment/Label', 'Confidence', 'Full article URL'])
        printed_summary = create_output_list(ticker_summary, ticker_score, cleaned_urls)
        
        print(printed_summary)
        print(articles)
        print(total_expense_dict)
        for analysis in printed_summary:
            st.subheader(f"{analysis[1]}")
            st.write(f"Sentiment: {analysis[2]}")
            st.write(f"Level of certainty/score: {100 * analysis[3]:.2f}%")
            st.write(f"Full article link: {analysis[4]}")
    except Exception as Err:
        st.write("An error has occured. Please give the model 20 seconds to load and try again")
        print(Err)
