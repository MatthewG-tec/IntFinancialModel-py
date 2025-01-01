import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from BlackScholes import black_scholes_call, black_scholes_put, black_scholes_call_div, black_scholes_put_div

def plot_normalized_prices(stock_data, stock_name, save_path=None):
    """
    Plots normalized monthly prices for the given stock.
    """
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Date'], stock_data['Normalized Price'], label=stock_name, color='blue', lw=2)
    plt.title(f"Normalized Stock Price for {stock_name}", fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Normalized Price', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return stock_data['Normalized Price'].describe()


def plot_paths(paths, save_path=None):
    """
    Plots the simulated stock price paths.
    Plots only the first 10 paths for clarity.
    
    Parameters:
    - paths (ndarray): Simulated stock price paths.
    - save_path (str or None): Path to save the plot. If None, the plot is displayed.
    """
    # Paths oriented correctly (num_paths * num_time_steps)
    if paths.shape[0] < paths.shape[1]:
        paths = paths.T
        
    plt.figure(figsize=(12, 8))  # Set figure size
    
    # Plot only the first 10 paths to avoid overcrowding
    for i, path in enumerate(paths[:100]):  
        plt.plot(path, alpha=0.7, linewidth=1)
    
    # Plot the initial price level
    initial_price = paths[0, 0]
    plt.axhline(initial_price, color='blue', linestyle='--', label=f'Initial Price: ${initial_price:.2f}')
    
    # Calculate and plot the expected price (mean of final prices)
    final_prices = paths[-1, :]
    expected_price = np.mean(final_prices)
    plt.axhline(expected_price, color='green', linestyle='--', label=f'Expected Price: ${expected_price:.2f}')
    
    # Add title and labels with improved font size and style
    plt.title('Simulated Stock Price Paths', fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Stock Price ($)', fontsize=14)
    
    # Add grid with dashed lines for readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend to the plot with key financial labels
    plt.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()  # Make sure everything fits nicely
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)  # High resolution
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
        
def plot_histogram(final_prices, bin_size=0.25, bar_width=0.7, save_path=None):
    """
    Plots the distribution of final stock prices as a histogram with key statistics in a legend.
    """
    # Compute histogram data
    bins = np.arange(min(final_prices), max(final_prices) + bin_size, bin_size)
    hist, edges = np.histogram(final_prices, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    
    # Compute key statistics
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    min_price = np.min(final_prices)
    max_price = np.max(final_prices)
    
    # Plot the histogram
    plt.figure(figsize=(12, 8))
    plt.bar(bin_centers, hist, width=bar_width, edgecolor='black', color='skyblue', alpha=0.7)
    
    # Add vertical lines for key statistics
    plt.axvline(mean_price, color='red', linestyle='--', label=f'Mean: ${mean_price:.2f}')
    plt.axvline(median_price, color='green', linestyle='--', label=f'Median: ${median_price:.2f}')
    plt.axvline(min_price, color='purple', linestyle='--', label=f'Min: ${min_price:.2f}')
    plt.axvline(max_price, color='orange', linestyle='--', label=f'Max: ${max_price:.2f}')
    
    # Add title and labels
    plt.title('Distribution of Final Stock Prices', fontsize=16)
    plt.xlabel('Stock Price ($)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add the legend
    plt.legend(loc='upper right', fontsize=12, frameon=True)
    
    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()

    # Return statistics
    return {"mean": mean_price, "median": median_price, "min": min_price, "max": max_price}



def plot_with_ITM_ATM_OTM(stock_name, stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, dividend_yield=0, save_path=None):
    """
    Plots the Black-Scholes option prices with ITM, ATM, and OTM regions.
    Uses the dividend-adjusted formulas if a dividend yield is provided.

    Parameters:
        - stock_name (str): Name of the stock.
        - stock_price (float): Current stock price.
        - strike_price (float): Strike price of the option.
        - time_to_maturity (float): Time to option expiration in years.
        - risk_free_rate (float): Annualized risk-free interest rate (e.g., 0.05 for 5%).
        - volatility (float): Annualized stock price volatility (e.g., 0.2 for 20%).
        - dividend_yield (float): Continuous dividend yield (optional, default=0).
        - save_path (str): Path to save the plot (optional, default=None).
    """
    # Generate a range of stock prices for plotting
    stock_prices = np.linspace(stock_price * 0.5, stock_price * 1.5, 100)

    # Compute option prices using the appropriate formula
    if dividend_yield > 0:
        call_prices = [black_scholes_call_div(s, strike_price, time_to_maturity, risk_free_rate, dividend_yield, volatility) for s in stock_prices]
        put_prices = [black_scholes_put_div(s, strike_price, time_to_maturity, risk_free_rate, dividend_yield, volatility) for s in stock_prices]
    else:
        call_prices = [black_scholes_call(s, strike_price, time_to_maturity, risk_free_rate, volatility) for s in stock_prices]
        put_prices = [black_scholes_put(s, strike_price, time_to_maturity, risk_free_rate, volatility) for s in stock_prices]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(stock_prices, call_prices, label='Call Option Price', color='green', lw=2)
    plt.plot(stock_prices, put_prices, label='Put Option Price', color='red', lw=2)

    # Highlight ITM/ATM/OTM regions
    plt.fill_between(stock_prices, 0, call_prices, where=(stock_prices >= strike_price), color='lightgreen', alpha=0.3, label='ITM (Call)')
    plt.fill_between(stock_prices, 0, put_prices, where=(stock_prices <= strike_price), color='lightcoral', alpha=0.3, label='ITM (Put)')
    plt.fill_between(stock_prices, 0, call_prices, where=(stock_prices < strike_price), color='lightblue', alpha=0.3, label='OTM (Call)')
    plt.fill_between(stock_prices, 0, put_prices, where=(stock_prices > strike_price), color='lightyellow', alpha=0.3, label='OTM (Put)')

    # Add vertical lines for key prices
    plt.axvline(x=strike_price, color='gray', linestyle='--', label='Strike Price (ATM)', lw=2)
    plt.axvline(x=stock_price, color='black', linestyle='--', label=f'Current Stock Price: {stock_price}', lw=2)

    # Add labels, title, and legend
    plt.title(f'{stock_name} Option Prices with ITM/ATM/OTM Regions', fontsize=16)
    plt.xlabel('Stock Price ($)', fontsize=14)
    plt.ylabel('Option Price ($)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

