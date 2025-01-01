# Main.py
import numpy as np
import pandas as pd
import json
import unittest
from ExcelParse import parse_sheets
from MonteCarloSim import MonteCarloSim
from BlackScholes import black_scholes_call, black_scholes_put, black_scholes_call_div, black_scholes_put_div
from Capm import CalcExpectedReturn, CalcBeta, Normalize, CalcMonthlyReturn
from Plotter import plot_normalized_prices, plot_paths, plot_histogram, plot_with_ITM_ATM_OTM
from AccuracyTest import test_capm_accuracy, test_monte_carlo_accuracy

# Import test cases
from Test import TestBlackScholes, TestCapm, TestMonteCarloSim

def save_json(data, filename="IntegratedModel.json"):
    """ Save calculated data to JSON file """
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print("=" * 80)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

def integrated_model(excel_file, capm_sheets, bs_sheet, mc_sheet):
    # Parse all required sheets from excel
    data = parse_sheets(excel_file, capm_sheets, bs_sheet, mc_sheet)
    results = {}

    # ---- CAPM Workflow ---- #
    print("=" * 80)
    print("CAPM ANALYSIS")
    print("=" * 80)
    capm_data = data["CAPM"]
    stock_data = capm_data["CAPM Sheet"]

    # Normalize and calculate returns
    normalized_data = Normalize(stock_data)
    stock_returns = CalcMonthlyReturn(stock_data)
    market_data = capm_data["CAPM Sheet"]
    market_returns = CalcMonthlyReturn(market_data)

    # Display normalized prices and returns (Only once)
    print("Normalized Prices and Monthly Returns:")
    combined_data = normalized_data[['Date', 'Price', 'Normalized Price']].copy()
    combined_data['Monthly Return'] = stock_returns['Monthly Return']  # Add Monthly Return to the dataframe
    print(combined_data.head())
    print("-" * 80)

    # Plot normalized prices
    plot_normalized_prices(normalized_data, stock_name="APPL-US")

    # Calculate beta and expected return
    beta = CalcBeta(stock_returns['Monthly Return'], market_returns['Monthly Return'])
    risk_free_rate = 0.0442  # Three Month U.S.A Treasury Bill
    market_return = 0.0990  # Expected Return S&P500
    expected_return = CalcExpectedReturn(risk_free_rate, beta, market_return)

    print(f"Beta: {beta:.4f}")
    print(f"Expected Return (CAPM): {expected_return:.4f}")

    results["CAPM"] = {
        "Beta": beta,
        "Expected Return (CAPM)": expected_return
    }

    # ---- Monte Carlo Simulation Workflow ---- #
    print("=" * 80)
    print("MONTE CARLO SIMULATION & GEOMETRIC BROWNIAN MOTION")
    print("=" * 80)

    mc_data = data["Monte Carlo"]
    mc_sim = MonteCarloSim(
        S0=mc_data["S0"], mu=expected_return, sigma=mc_data["sigma"], 
        T=mc_data["T"], num_simulations=mc_data["num_simulations"], num_steps=mc_data["num_steps"]
    )
    simulated_paths = mc_sim.simulate_paths()
    expected_final_price = mc_sim.calc_expected_final_price(simulated_paths)
    mc_volatility = mc_sim.calc_volatility_from_paths(simulated_paths)

    # Sample paths
    print("Sample of Simulated Price Paths (First 5):")
    for i, path in enumerate(simulated_paths[:5]):
        print(f"  Path {i+1}: [{', '.join(f'{p:.2f}' for p in path[:3])}, ..., {', '.join(f'{p:.2f}' for p in path[-3:])}]")

    # Plot sample paths
    plot_paths(simulated_paths)

    # Plot histogram of final prices
    final_prices = simulated_paths[:, -1]
    plot_histogram(final_prices)

    # Summary statistics
    print("-" * 80)
    print("Expected Final Price from MC Simulation:", expected_final_price)
    print("Volatility from MC Simulation:", mc_volatility)

    results["Monte Carlo"] = {
        "Expected Final Price": expected_final_price,
        "Volatility": mc_volatility
    }
    
    # Extract historical prices for Monte Carlo testing
    historical_prices_df = capm_data["CAPM Sheet"].copy()
    historical_prices_df['Date'] = pd.to_datetime(historical_prices_df['Date'])  # Ensure 'Date' is a datetime object
    historical_prices_df = historical_prices_df.sort_values(by='Date', ascending=False).head(50)

    # ---- Black-Scholes Workflow ---- #
    print("=" * 80)
    print("BLACK-SCHOLES OPTION PRICING")
    print("=" * 80)

    bs_data = data["Black-Scholes"]
    stock_price = bs_data["Stock Price"]
    strike_price = bs_data["Strike Price"]
    time_to_maturity = bs_data["Time to Maturity"]
    risk_free_rate = bs_data["Risk-Free Rate"]
    volatility = bs_data["Volatility"]
    dividend_yield = bs_data["Dividend Yield"]

    call_price = black_scholes_call(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    put_price = black_scholes_put(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    call_price_div = black_scholes_call_div(stock_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, volatility)
    put_price_div = black_scholes_put_div(stock_price, strike_price, time_to_maturity, risk_free_rate, dividend_yield, volatility)

    print(f"Call Option Price: {call_price:.4f}")
    print(f"Put Option Price: {put_price:.4f}")
    print(f"Call Option Price with Dividend Yield: {call_price_div:.4f}")
    print(f"Put Option Price with Dividend Yield: {put_price_div:.4f}")

    # Plot option prices with ITM, ATM, OTM regions
    plot_with_ITM_ATM_OTM(stock_name="APPL-US", stock_price=stock_price, strike_price=strike_price,
                          time_to_maturity=time_to_maturity, risk_free_rate=risk_free_rate, volatility=volatility, 
                          dividend_yield=dividend_yield)

    results["Black-Scholes"] = {
        "Call Price": call_price,
        "Put Price": put_price,
        "Call Price with Dividends": call_price_div,
        "Put Price with Dividends": put_price_div
    }

    # ---- Accuracy Tests ---- #
    print("=" * 80)
    print("ACCURACY TESTS")
    print("=" * 80)

    # Run accuracy tests
    capm_mape = test_capm_accuracy(data, risk_free_rate, market_return, beta)
    mc_rmse = test_monte_carlo_accuracy(simulated_paths, data, historical_prices_df)
    print(f"CAPM MAPE: {capm_mape:.2f}%")
    print(f"Monte Carlo RMSE: ${mc_rmse:.2f}")

    # Add accuracy results to the final results
    results["Accuracy"] = {
            "CAPM MAPE": capm_mape,
            "Monte Carlo RMSE": mc_rmse
    }

    # ---- Save Results to JSON ---- #
    save_json(results)

    return results

def main():
    # Define the Excel file and sheets
    excel_file = "Database.xlsx"
    capm_sheets = ["CAPM Sheet"]
    bs_sheet = "Black Scholes Sheet"
    mc_sheet = "Monte Carlo Sheet"
    
    # Run the integrated model
    results = integrated_model(excel_file, capm_sheets, bs_sheet, mc_sheet)

    # Display Final Results
    print("=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    for model_name, model_results in results.items():
        print(f"\n{model_name} Results:")
        for key, value in model_results.items():
            print(f"  {key}: {value:.2f}")

    # ---- Running Tests ---- #
    print("=" * 80)
    print("Running Tests...")
    print("=" * 80)

    # Collect test results
    test_suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestBlackScholes),
        unittest.TestLoader().loadTestsFromTestCase(TestCapm),
        unittest.TestLoader().loadTestsFromTestCase(TestMonteCarloSim)
    ]
    all_tests = unittest.TestSuite(test_suites)

    # Suppress detailed output and store results
    with open("test_results.txt", "w") as f:
        test_runner = unittest.TextTestRunner(stream=f, verbosity=0)
        test_result = test_runner.run(all_tests)

    # Print the results summary at the end
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run: {test_result.testsRun}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Failures: {len(test_result.failures)}")
    if test_result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed or had errors.")
    print("=" * 80)


if __name__ == "__main__":
    main()
  
