# Quantitative Financial Analysis and Comparison Using Python

**Author**: Matthew Gillett

## Integrated Financial Model

### Compiling Code

- **Spyder IDE**:
 1. Make sure the Excel file is not open when running.
 2. Compile by using the "Run File" button (green play button).
- **Terminal**:
 1. Run: `python main.py`
 2. This runs the tests and everything from executing `main.py`.

### Python Files

- `MonteCarloSim.py`
- `BlackScholes.py`
- `Capm.py`
- `Plotter.py`
- `ExcelParse.py`
- `AccuracyTest.py`
- `Test.py`

### Required Libraries

- `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`, `unittest`

You can install these libraries using:
```bash
pip install numpy pandas scipy matplotlib openpyxl unittest
```

### Data
- Data is pulled from the FactSet database.
- Access it through the Dhillon School of Business student account.

### Order of Operaitons
1. CAPM --> Price Path Simulation (GBM)
    - Expected Return (E(Ri)):
        - The CAPM model provides the stock's expected return, which becomes the drift in the GBM formula.
    - Volatility:
        - While CAPM does not calculate volatility directly, it helps justify assumptions for consistency.
2. Price Path Simulation (GBM) --> Monte Carlo Simulation
    - Simulated Price Paths (S(t)):
        - The GBM formula generates multiple price paths over time, considering the drift and volatility.
    - Monte Carlo Simulation:
        - Price paths are used to create a distribution of possible outcomes, enabling probability-based calculations.
    - Key Metrics Passed on:
        - Expected future price (E(St)).
        - Distribution of prices at specific times (St).
        - Range and probabilities for price thresholds.
3. Monte Carlo Simulations → Black-Scholes
    - Simulated Price Paths (S(T)):
        - Monte Carlo provides a distribution of future prices, which can help validate or replace Black-Scholes assumptions.
    - Expected Terminal Price (E(St)):
        - The mean simulated terminal price may be used as a proxy for the current stock price input (S0) in the Black-Scholes formula.
    - Volatility:
        - Black-Scholes requires a consistent volatility measure. The Monte Carlo Simulation can refine this input by analyzing the simulated price paths.
4. Black-Scholes → Decision-Making
    - Option Prices:
        - Black-Scholes provides the value of options based on the underlying asset's price dynamics.
    - Risk Metrics:
        - Derive the delta (rate of change of option price concerning stock price).
        - Use simulated price paths to assess sensitivities and validate Black-Scholes assumptions.
