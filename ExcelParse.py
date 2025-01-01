# ExcelParse.py

import pandas as pd

def parse_capm_sheet(excel_file, capm_sheet):
    """
    Parse all sheet related to CAPM calculations.
    
    Parameters:
        excel_file (str): Path to excel file.
        capm_sheet (list): List of sheet names for CAPM.
    
    Returns:
        dict: Parsed CAPM data for each sheet.
    """
    capm_data = {}
    
    for sheet in capm_sheet:
        df = pd.read_excel(excel_file, sheet_name=sheet, header=1)

        # Strip any leading/trailing spaces in column names
        df.columns = df.columns.str.strip()

        # Check if 'Date' and 'Price' columns exist
        if 'Date' in df.columns and 'Price' in df.columns:
            # Make sure 'Date' column is in correct datetime format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Filter out rows where Date or Price is NaN
            df = df.dropna(subset=['Date', 'Price'])
            # Convert 'Price' to numeric if not already
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            
            # Store relevant data
            capm_data[sheet] = df[['Date', 'Price']]

        else:
            raise KeyError(f"The 'Date' or 'Price' column is missing in the sheet '{sheet}'.")

    return capm_data

def parse_monte_carlo_sheet(excel_file, mc_sheet):
    """
    Parse sheet related to Monte Carlo simulations.
    
    Parameters:
        excel_file (str): Path to the Excel file.
        mc_sheet (str): Sheet name for Monte Carlo simulation.
    
    Returns:
        dict: Parsed Monte Carlo parameters.
    """
    df = pd.read_excel(excel_file, sheet_name=mc_sheet, header = 1)
    
    # Extract Monte Carlo parameters
    initial_price = df.loc[0, 'Initial Price']
    expected_return = df.loc[0, 'Expected Return']
    volatility = df.loc[0, 'Volatility']
    time_period = df.loc[0, 'Time Period']
    simulations = int(df.loc[0, 'Simulations'])
    steps = int(df.loc[0, 'Steps'])
    
    return {
        'S0': initial_price,
        'mu': expected_return,
        'sigma': volatility,
        'T': time_period,
        'num_simulations': simulations,
        'num_steps': steps
    }

def parse_black_scholes_sheet(excel_file, bs_sheet):
    """
    Parse sheet related to Black-Scholes calculations.
    
    Parameters:
        excel_file (str): Path to the Excel file.
        bs_sheet (str): Sheet name for Black-Scholes.
    
    Returns:
        dict: Parsed Black-Scholes parameters.
    """
    df = pd.read_excel(excel_file, sheet_name=bs_sheet, header = 1)
    
    # Extract Black-Scholes parameters
    stock_price = df.loc[0, 'Stock Price']
    strike_price = df.loc[0, 'Strike Price']
    time_to_maturity = df.loc[0, 'Time to Maturity']
    risk_free_rate = df.loc[0, 'Risk-Free Rate']
    volatility = df.loc[0, 'Volatility']
    dividend_yield = df.loc[0, 'Dividend Yield'] if 'Dividend Yield' in df.columns else 0
    
    return {
        'Stock Price': stock_price,
        'Strike Price': strike_price,
        'Time to Maturity': time_to_maturity,
        'Risk-Free Rate': risk_free_rate,
        'Volatility': volatility,
        'Dividend Yield': dividend_yield
    }

def parse_sheets(excel_file, capm_sheets, bs_sheet, mc_sheet):
    """
    Parse all required sheets for CAPM, Black-Scholes, and Monte Carlo Simulation.
    
    Parameters:
        excel_file (str): Path to the Excel file.
        capm_sheets (list): List of sheet names for CAPM.
        bs_sheet (str): Sheet name for Black-Scholes.
        mc_sheet (str): Sheet name for Monte Carlo Simulation.
    
    Returns:
        dict: Consolidated data for all models.
    """
    capm_data = parse_capm_sheet(excel_file, capm_sheets)
    black_scholes_data = parse_black_scholes_sheet(excel_file, bs_sheet)
    monte_carlo_data = parse_monte_carlo_sheet(excel_file, mc_sheet)
    
    return {
        'CAPM': capm_data,
        'Black-Scholes': black_scholes_data,
        'Monte Carlo': monte_carlo_data
    }
