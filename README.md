# Multi Equity Analyses Streamlit Application

This application allows users to perform various analyses for provided Ticker symbol(s). 
The first section of the application allows users to perform Monte Carlo simulations on the desired ticker symbol's adjusted closing price concluding with values of the most likely and the least likely prices for a given period. 


The second section of the application performs Portfolio optimization using Monte Carlo simulations on random walks for a 125,000 scenarios. This section returns a visualization of the Efficient Frontiers along with the ideal allocations of the provided ticker symbols to create a portfolio that maximizes return while minimizing the risks. 


The third section of the appplication works by retrieving the latest news on the desired Ticker and returning the links to the articles. It the proceeds to read the articles from the provided links and provide a summary with sentiment analysis to measure whether the news are positive, negative or neutral for the company in question. It uses various methods such as a HuggingFace Financial Summarization and Text Classification NLP Models, Monte Carlo simulations and Markowitz Efficient Frontier Theory.
