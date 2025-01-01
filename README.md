Quantitative Financial Analysis and Comparison Using Python
Author: Matthew Gillett
--------------------------------------------------------------
Integrated Finanical Model:
--------------------------------------------------------------
Compiling Code:
    - Spyder IDE:
        1. Make sure excel file is not open when running**
        2. Compile by using run file (green play button)
    - Terminal:
        1. python main.py
        2. Runs tests and everything from running main.py
--------------------------------------------------------------
MonteCarloSim.py
BlackScholes.py
Capm.py
Plotter.py
ExcelParse.py
AccuracyTest.py
Test.py
--------------------------------------------------------------
Required Libraries:
    - numpy, pandas, scipy, matplotlib, openpyxl, unittest
    - Can use "pip install ..."
--------------------------------------------------------------
Data:
    - Pulled from the FactSet data base.
    - Access through dhillon school of business student account.
--------------------------------------------------------------
** Does run on a standard terminal if python is installed and pip installed libraries
** If you want to run on IDE I used https://www.spyder-ide.org/ comes standard with all financial libraries
--------------------------------------------------------------
ORDER OF OPERATIONS

CAPM --> Price PAth Simulation (GBM):
    1. Expected Return (E(Ri))
        - CAPM model provides stocks expected return, which becomes drift in the GBM formula.
    2. Volatility
        - While CAPM does not calculate volatility directly, it helps justify assumptions for consistency.

Price Path Simulation --> Monte Carlo Simulations:
    1. Simulated Price Paths (S(t))
        - The GBM formula generates multiple price paths over time, considering the drift and volatility.
    2. Monte Carlo Simulation
        - Price paths are used to create a distribution of possible outcomes, enabling probability
        based calculations.
    3. Key Metrics Passed on
        - Expected future price (E(St)).
        - Distribution of prices at specific time (St).
        - Range and probabilities for price thresholds. 
        
Monte Carlo Simulations --> Black Scholes:
    1. Simulated Price Paths (S(T))
        - Monte Carlo provides a distributon of future prices, which can help validate or replace Black-Scholes assumptions. 
    2. Expected Terminal Price (E(St))
        - the mean simulated terminal price may be used as a proxy for the current stock price input (S0) in the Black-Scholes. 
    3. Volatility
        - Black-Scholes requires a consistent volatility measure. The Monte Carlo Simulation can refine this input by analyzing the simulated
        price paths. 

Black-Scholes --> Decision-Making:
    1. Option Prices
        - Black Scholes provides the value of options based on the underlying assets price dynamics.
    2. Risk Metrics
        - Derive the delta (rate of change of option price concerning stock price).
        - Use simulated price paths to assess sensitivites and validate Black-Scholes assumptions.
--------------------------------------------------------------
            
