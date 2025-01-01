# MonteCarloSim.py
import numpy as np

class MonteCarloSim:
    def __init__(self, S0, mu, sigma, T, num_simulations, num_steps):
        self.S0 = S0  # Initial stock price
        self.mu = mu  # Drift (expected return, typically from CAPM)
        self.sigma = sigma  # Volatility (std dev of returns)
        self.T = T  # Time period (in years)
        self.num_simulations = num_simulations  # Number of simulated paths
        self.num_steps = num_steps  # Number of time steps
        self.dt = T / num_steps  # Time step size
        
    def simulate_paths(self):
        # Initialize an array to store price paths
        paths = np.zeros((self.num_steps, self.num_simulations))
        paths[0] = self.S0  # Set the initial stock price for all paths
        
        # Simulate each path
        for t in range(1, self.num_steps):
            Z = np.random.standard_normal(self.num_simulations)  # Generate random values for Brownian Motion (Z ~ N(0, 1))
            # Update the price for the next time step using GBM
            paths[t] = paths[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)
        
        # Remove outliers from simulated paths
        paths = self.remove_outliers_from_paths(paths)
        
        return paths

    def calc_expected_final_price(self, paths):
        # Calculate the expected final price (mean of the paths)
        expected_price = np.mean(paths[-1, :])
        return expected_price

    def calc_volatility_from_paths(self, paths):
        # Calculate volatility from simulated paths (std dev of log returns)
        log_returns = np.diff(np.log(paths), axis=0)
        return np.std(log_returns)

    def remove_outliers_from_paths(self, paths, threshold=3):
        """ Remove outliers from simulated paths using Z-score method """
        # Calculate mean and standard deviation along the time axis (axis 0)
        mean_paths = np.mean(paths, axis=0)
        std_paths = np.std(paths, axis=0)
        
        # Calculate Z-scores for each simulation
        z_scores = np.abs((paths - mean_paths) / std_paths)
        
        # Create a boolean mask for simulations that are within the Z-score threshold
        valid_simulations = np.all(z_scores < threshold, axis=0)
        
        # Filter the paths using the boolean mask
        paths = paths[:, valid_simulations]
        
        return paths
      
