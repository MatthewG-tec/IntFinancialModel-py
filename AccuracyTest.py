# AccuracyTest.py
import numpy as np
import pandas as pd
import json
from MonteCarloSim import MonteCarloSim
from Capm import CalcExpectedReturn, CalcMonthlyReturn, CalcBeta
from ExcelParse import parse_sheets

def calc_mape(actual_returns, expected_returns):
    """Calculate the Mean Absolute Percentage Error."""
    actual_returns = np.array(actual_returns)  # Convert to numpy array
    expected_returns = np.array(expected_returns)  # Convert to numpy array
    return np.mean(np.abs((actual_returns - expected_returns) / actual_returns)) * 100

def test_capm_accuracy(data, rf, market_return, beta):
    """
    Test the accuracy of CAPM by comparing expected returns to actual returns.
    Arguments: ** use up to 23 and then compare to the most present expected returns** 
        data (dict): Parsed stock data.
        rf (float): Risk-free rate.
        market_return (float): Expected market return.
        beta (float): Calculated beta value for the stock.
    Returns:
        float: The MAPE of the CAPM model.
    """
    capm_data = data["CAPM"]
    stock_data = capm_data["CAPM Sheet"]
    
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    stock_data = stock_data.sort_values('Date')
    
    # Calculate time span in years
    time_span = (stock_data['Date'].max() - stock_data['Date'].min()).days / 365.25
    
    # Get first and last prices
    start_price = stock_data['Price'].iloc[0]
    end_price = stock_data['Price'].iloc[-1]
    
    # Calculate Compound Annual Growth Rate
    annualized_actual_returns = (end_price / start_price) ** (1 / time_span) - 1
    
    # Calculate the expected annual return using CAPM
    expected_annual_return = CalcExpectedReturn(rf, beta, market_return)
    
    print(f"Start Date: {stock_data['Date'].iloc[0]}")
    print(f"End Date: {stock_data['Date'].iloc[-1]}")
    print(f"Time Span: {time_span:.2f} years")
    print(f"Start Price: ${start_price:.2f}")
    print(f"End Price: ${end_price:.2f}")
    print(f"Annualized Actual Returns: {annualized_actual_returns:.4f} or {annualized_actual_returns*100:.2f}%")
    print(f"CAPM Expected Annual Return: {expected_annual_return:.4f} or {expected_annual_return*100:.2f}%")
    
    mape = calc_mape([annualized_actual_returns], [expected_annual_return])
    return mape

def calculate_rmse(actual_prices, simulated_prices):
    # Ensure the lengths match
    min_length = min(len(actual_prices), len(simulated_prices))
    actual_prices = actual_prices[:min_length]
    simulated_prices = simulated_prices[:min_length]
    
    return np.sqrt(np.mean((actual_prices - simulated_prices) ** 2))

def test_monte_carlo_accuracy(simulated_paths, data, historical_prices_df):
    """
    Test the accuracy of Monte Carlo simulations by comparing simulated and actual prices.
    Arguments:
        simulated_paths (ndarray): Simulated price paths (num_steps x num_simulations).
        data (dict): Parsed data dictionary from parse_sheets().
        historical_prices_df (DataFrame): DataFrame containing historical prices (date and price).
    Returns:
        float: The RMSE of the Monte Carlo model.
    """
    # Extract the CAPM data
    capm_data = data['CAPM']
    
    capm_sheet = list(capm_data.values())[0]
    
    if 'Price' not in capm_sheet.columns:
        raise KeyError("The 'Price' column is missing in the actual prices data.")

    actual_prices = capm_sheet['Price'].values
    
    historical_prices_df['Date'] = pd.to_datetime(historical_prices_df['Date'])

    historical_prices_df = historical_prices_df.sort_values(by='Date', ascending=False).head(5)

    # Extract the actual prices for the most recent 50 entries (ensure dates match between CAPM and historical)
    recent_prices = historical_prices_df['Price'].values

    simulated_final_prices = simulated_paths[-1, :]

    min_length = min(len(recent_prices), len(simulated_final_prices))
    actual_final_prices = recent_prices[:min_length]
    simulated_final_prices = simulated_final_prices[:min_length]

    rmse = np.sqrt(np.mean((actual_final_prices - simulated_final_prices) ** 2))
    
    return rmse

def run_integrated_model_accuracy_test(excel_file, capm_sheets, bs_sheet, mc_sheet, rf, market_return, beta, option_data):
    # Parse data from Excel
    data = parse_sheets(excel_file, capm_sheets, bs_sheet, mc_sheet)

    # Test CAPM accuracy
    capm_mape = test_capm_accuracy(data, rf, market_return, beta)

    # Parse Monte Carlo parameters
    monte_carlo_params = data['Monte Carlo']

    # Test Monte Carlo accuracy
    mc_rmse = test_monte_carlo_accuracy(data, monte_carlo_params)

    # Return all the results for further analysis
    return {
        'CAPM MAPE': capm_mape,
        'Monte Carlo RMSE': mc_rmse
    }
