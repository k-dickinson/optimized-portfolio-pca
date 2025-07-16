import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov


tickers = ["INTC", "ORLA", "RTX", "CMG", "XOM"]
data = yf.download(tickers, start="2022-01-01", end="2024-01-01", auto_adjust=False)

# If multiple tickers: columns are MultiIndex like ('Adj Close', 'AAPL')
# So we grab only the 'Adj Close' level
adj_close = data['Adj Close']



#percent change
returns = data.pct_change().dropna()

#expected returns
mean_returns = returns.mean() * 252  # annualized
#covariance matrix
cov_matrix = returns.cov() * 252     # annualized

#average historical return for each asset
mu = mean_historical_return(data)
#how asset returns vary with each other (volatility and correlations)
S = sample_cov(data)

#initialize the Efficient Frontier optimizer with expected returns and risk
ef = EfficientFrontier(mu, S)
#find the portfolio weights that maximize the Sharpe ratio
weights = ef.max_sharpe()  # or ef.min_volatility()
#removes very small weights and rounds values for better readability
cleaned_weights = ef.clean_weights()
#prints the expected return, volatility, and Sharpe ratio of the optimized portfolio
ef.portfolio_performance(verbose=True)