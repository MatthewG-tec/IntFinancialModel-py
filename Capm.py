# Capm.py
import numpy as np

def remove_outliers(df, column_name='Price', method='z-score', threshold=3):
    """ Remove outliers using Z-score or IQR method """
    if method == 'z-score':
        z_scores = (df[column_name] - df[column_name].mean()) / df[column_name].std()
        df = df[(np.abs(z_scores) < threshold)]  # Keep rows where z-score is within threshold
    
    elif method == 'iqr':
        # Calculate IQR (Interquartile Range)
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column_name] >= (Q1 - 1.5 * IQR)) & (df[column_name] <= (Q3 + 1.5 * IQR))]

    return df

def CalcExpectedReturn(rf, beta, market_return):
    """ 
    Calculate expected return using CAPM formula.
    Returns:
        float: Expected return based on CAPM formula.
    """
    return rf + beta * (market_return - rf)

def Normalize(df):
    """
    Normalize stock data based on the initial price.
    Returns:
        DataFrame with normalized stock prices
    """
    df = remove_outliers(df)
    df['Normalized Price'] = df['Price'] / df['Price'].iloc[0]
    return df

def CalcMonthlyReturn(df):
    """
    Calculate monthly returns.
    Returns:
        DataFrame with monthly returns added
    """
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Sort by the 'Date' column
    df = df.sort_values(by='Date')
    
    # Calculate monthly returns based on 'Price'
    df['Monthly Return'] = df['Price'].pct_change().fillna(0)
    
    return df

def CalcBeta(stock_returns, market_returns):
    """
    Calculate beta between stock and market returns.    
    Returns:
        float: Calculated beta of the stock relative to the market
    """
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta
