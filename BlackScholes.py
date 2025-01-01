# BlackScholes.py
"""
VARIABLES:
    - S: Current asset price.
    - K: Strike price of option.
    - r: Risk-free rate.
    - T: Time till expiration.
    - q: continuous dividend yield
    - Sigma: Annualized volatility of asset's returns

DISTRIBUTION:
    - cumulative distribution function for standard normal distribution:    
"""

# blackscholes.py
import math
from scipy.stats import norm
"""
FORMULA WITH OUT DIVIDEND:
    - Call = S0N(d1) - N(d2)Ke^-rT
    - Put = N(-d2)ke^-rT - N(-d1)S0
    - d1 = [ln(S/K) + (r + (sigma^2/2))T]/[sigma(root(T))]
    - d2 = d1 - sigma(root(T))
"""
def black_scholes_call(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    # Calculate d1 and d2
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    
    # Calculate call price
    call_price = stock_price * norm.cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    return call_price

def black_scholes_put(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    # Calculate d1 and d2
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    
    # Calculate put price
    put_price = strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
    return put_price

"""
FORMULA WITH DIVIDEND:
    - Call = S0e^-qT(N(d1)) - Ke^-rT(N(d2))
    - Put = Ke^-rT(N(-d2)) - S0e^-qT(N(-d1))
    - d1 = [ln(S/K) + (r - q + (1/2(sigma^2)T))]/[sigma(root(T))]
    - d2 = d1 - sigma(root(T))
"""
def black_scholes_call_div(stock_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, volatility):
    # Calculate d1 and d2 with dividend adjustment
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    
    # Call option price formula with dividends
    call_price_div = stock_price * math.exp(-dividend_yield * time_to_maturity) * norm.cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    return call_price_div

def black_scholes_put_div(stock_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, volatility):
    # Calculate d1 and d2 with dividend adjustment
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    
    # Put option price formula with dividends
    put_price_div = strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - stock_price * math.exp(-dividend_yield * time_to_maturity) * norm.cdf(-d1)
    return put_price_div
