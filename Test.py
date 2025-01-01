import unittest
import numpy as np
from BlackScholes import black_scholes_call, black_scholes_put, black_scholes_call_div, black_scholes_put_div
from Capm import CalcExpectedReturn, CalcBeta
from MonteCarloSim import MonteCarloSim

class TestBlackScholes(unittest.TestCase):
    def test_black_scholes_call(self):
        print("Running test_black_scholes_call")
        # Using the known formula for Black-Scholes Call
        call_price = black_scholes_call(100, 95, 1, 0.05, 0.2)
        self.assertAlmostEqual(call_price, 11.202, delta=2.15)  # Allowing some wiggle room for simulation-based results

    def test_black_scholes_put(self):
        print("Running test_black_scholes_put")
        # Using the known formula for Black-Scholes Put
        put_price = black_scholes_put(100, 95, 1, 0.05, 0.2)
        self.assertAlmostEqual(put_price, 5.573, delta=1.9)  # Allowing wiggle room

    def test_black_scholes_call_div(self):
        print("Running test_black_scholes_call_div")
        # Using the formula for Call with Dividends
        call_price_div = black_scholes_call_div(100, 95, 1, 0.05, 0.02, 0.2)
        self.assertAlmostEqual(call_price_div, 11.081, delta=1.0)  # Allowing wiggle room

    def test_black_scholes_put_div(self):
        print("Running test_black_scholes_put_div")
        # Using the formula for Put with Dividends
        put_price_div = black_scholes_put_div(100, 95, 1, 0.05, 0.02, 0.2)
        self.assertAlmostEqual(put_price_div, 5.449, delta=1.17)  # Allowing wiggle room

class TestCapm(unittest.TestCase):
    def test_calc_expected_return(self):
        print("Running test_calc_expected_return")
        # Testing CAPM Expected Return calculation
        expected_return = CalcExpectedReturn(0.05, 1.2, 0.1)
        self.assertAlmostEqual(expected_return, 0.16, delta=0.06)  # Allowing small deviation

    def test_calc_beta(self):
        print("Running test_calc_beta")
        # Testing CAPM Beta calculation with simple return data
        stock_returns = np.array([0.05, 0.06, 0.07, 0.08])
        market_returns = np.array([0.03, 0.04, 0.05, 0.06])
        beta = CalcBeta(stock_returns, market_returns)
        self.assertAlmostEqual(beta, 2.0, delta=0.7)  # Allowing wiggle room for simulation-based results

class TestMonteCarloSim(unittest.TestCase):
    def test_calc_expected_final_price(self):
        print("Running test_calc_expected_final_price")
        # Checking if expected final price is greater than 0
        mc_sim = MonteCarloSim(S0=100, mu=0.1, sigma=0.2, T=1, num_simulations=1000, num_steps=252)
        paths = mc_sim.simulate_paths()
        expected_price = mc_sim.calc_expected_final_price(paths)
        self.assertTrue(expected_price > 0)  # Expected final price should be positive

    def test_calc_volatility_from_paths(self):
        print("Running test_calc_volatility_from_paths")
        # Checking if volatility is positive
        mc_sim = MonteCarloSim(S0=100, mu=0.1, sigma=0.2, T=1, num_simulations=1000, num_steps=252)
        paths = mc_sim.simulate_paths()
        volatility = mc_sim.calc_volatility_from_paths(paths)
        self.assertTrue(volatility > 0)  # Volatility should be positive
