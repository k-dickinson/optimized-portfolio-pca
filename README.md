# Portfolio Optimization and PCA Factor Analysis on Stock Returns

This project demonstrates a workflow combining portfolio optimization and principal component analysis (PCA) to analyze stock returns. Additionally, it explores correlations of PCA factors with sector ETFs to interpret underlying market drivers.

---

## Overview

- **Portfolio Optimization:** Using historical price data, we calculate expected returns and covariance to construct a minimum volatility portfolio with constrained weights.

- **Principal Component Analysis (PCA):** We reduce the dimensionality of stock return data to identify dominant patterns (factors) explaining the majority of variance.

- **Factor Interpretation:** We compute PCA factor returns and correlate them with sector ETFs to interpret each factor’s economic meaning.

---

## Table of Contents

- [Requirements](#requirements)  
- [Data Acquisition](#data-acquisition)  
- [Portfolio Optimization](#portfolio-optimization)  
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)  
- [Factor Returns](#factor-returns)  
- [Factor-ETF Correlation Heatmap](#factor-etf-correlation-heatmap)  
- [Visualizations](#visualizations)  
- [Interpretation and Limitations](#interpretation-and-limitations)  
- [How to Run](#how-to-run)  
- [References](#references)  

---

## Requirements

- Python 3.7+

- Libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `PyPortfolioOpt`

Install dependencies with:

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn PyPortfolioOpt
```

## Data Acquisition

We fetch adjusted close prices for a selected basket of stocks over a specified historical period. These prices are used to compute daily returns, which are the basis for both portfolio optimization and PCA.

Example stocks:

```python
tickers = ["INTC", "ORLA", "RTX", "CMG", "XOM", "GOOG"]
```

We use Yahoo Finance via `yfinance` for data retrieval, which is a convenient API to get historical stock data.

---

## Portfolio Optimization

Portfolio optimization aims to select the best combination of asset weights to minimize risk or maximize return based on historical data.

### Key Concepts

- **Expected Returns (mu):** The average or expected return of each asset, often estimated by historical mean returns.

- **Covariance Matrix (S):** Measures how asset returns move together, representing portfolio risk.

- **Weight Bounds:** Constraints on the minimum and maximum allocations for each asset (e.g., between 5% and 60%).

- **Minimum Volatility Portfolio:** The portfolio configuration with the lowest overall volatility (risk), meeting constraints.

### Implementation

Using PyPortfolioOpt, we calculate expected returns and covariance matrix from historical adjusted close prices, then find the minimum volatility portfolio weights.

**Full portfolio optimization code [here](put the link here)**


## Principal Component Analysis (PCA)

PCA reduces the dimensionality of stock return data by transforming correlated variables into uncorrelated principal components (PCs) that capture the major sources of variance.

### Why PCA on Stock Returns?

Stock returns often share underlying factors such as market movements or sector trends. PCA helps:

- Identify these latent factors.
- Simplify the data.
- Reduce noise.
- Aid in factor-based portfolio strategies.

### How PCA Works

- Finds orthogonal directions (PCs) that explain decreasing amounts of variance.
- The explained variance ratio quantifies how much variance each PC explains.
- Loadings (component weights) show how each original stock contributes to each PC.

---

## Factor Returns

Factor returns are the projections of the original stock returns onto the PCA components, producing time series that represent each latent factor’s daily return.

These factor returns can be used for further analysis and interpretation.

---

## Factor-ETF Correlation Heatmap

To interpret PCA factors economically, we correlate factor returns with sector ETFs such as:

- Energy (XLE)
- Technology (XLK)
- Financials (XLF)
- Broad market (SPY, VTI)
- Nasdaq tech-heavy index (QQQ)

Correlation coefficients are visualized as a heatmap to help identify which sectors correspond to each factor.

---

## Visualizations

We generate the following visual components:

- Scree Plot: Shows how much variance each PC explains.
- PCA Component Loadings Table: Shows stock contributions to each factor.
- Portfolio Summary: Sharpe ratio, expected return, and optimized weights.
- Factor Returns Over Time: Line plots for the first few PCs.
- Interpretation Text Box: Explains factor economic meanings.
- Factor-ETF Correlation Heatmap: Visualizes factor-sector relationships.

Full code [here](put the link here)

## Interpretation and Limitations

Based on the heatmap correlations:

- **PC1:** Likely a broad market factor (general market movements).
- **PC2:** Likely represents the tech sector.
- **PC3:** Likely corresponds to the energy sector.

**Note on nonlinearities:**  
This PCA and correlation analysis assumes linear relationships between returns and factors. Nonlinear dependencies are not captured, and thus the heatmap interpretation might miss complex interactions. Handling nonlinearities requires more advanced methods and is beyond this project’s scope.

---

## How to Run

1. Clone this repository.
2. Install required packages.
3. Insert your portfolio optimization code and visualization code in the marked sections.
4. Run the script.
5. Analyze visual outputs and interpretation text.

---

## References

- [PyPortfolioOpt documentation](https://pyportfolioopt.readthedocs.io/en/latest/)
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Yahoo Finance (yfinance)](https://pypi.org/project/yfinance/)
- [Seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
