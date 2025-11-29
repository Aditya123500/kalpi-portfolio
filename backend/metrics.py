import numpy as np
import pandas as pd

def calculate_portfolio_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, risk_free_rate: float = 0.0, mar: float = 0.0):
    """
    Calculates metrics with correct Risk Scaling (Sqrt Time) and Robust Alignment.
    """
    # Annualize Mean Return (Linear) and Volatility (Square Root Rule)
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    
    # 1. Sharpe Ratio: Risk-adjusted return above Risk-Free Rate
    sharpe = (mu - risk_free_rate) / sigma if sigma > 0 else 0
    
    # 2. Sortino Ratio: Similar to Sharpe but penalizes only downside volatility (bad risk)
    downside_returns = returns[returns < mar]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (mu - mar) / downside_std if downside_std > 0 else 0
    
    # 3. Omega Ratio: Probability weighted ratio of gains vs losses above MAR threshold
    excess_ret = returns - (mar / 252)
    positive_sum = excess_ret[excess_ret > 0].sum()
    negative_sum = abs(excess_ret[excess_ret < 0].sum())
    omega = positive_sum / negative_sum if negative_sum > 0 else 0
    
    # 4. CVaR (95%): Expected loss on the worst 5% of days
    if len(returns) > 0:
        var_95 = np.percentile(returns, 5) # Historical 5% VaR
        # Annualized CVaR: Average loss below VaR cutoff * sqrt(time) scaling
        cvar_95 = returns[returns <= var_95].mean() * np.sqrt(252)
    else:
        cvar_95 = 0.0
    
    # 5. Max Drawdown: Largest peak-to-trough decline
    cum_returns = (1 + returns).cumprod()
    if not cum_returns.empty:
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_dd = drawdown.min()
    else:
        max_dd = 0.0

    # 6. Tracking Error & Information Ratio (Active Management Metrics)
    te = 0.0
    ir = 0.0
    
    # Robust Alignment: Ensure benchmark exists and matches dates before calculating
    if benchmark_returns is not None:
        try:
            # Normalize input: Ensure we have a Series, not a DataFrame
            if isinstance(benchmark_returns, pd.DataFrame):
                bench_series = benchmark_returns.iloc[:, 0]
            else:
                bench_series = benchmark_returns

            # Arithmetic Alignment: Subtraction automatically aligns indices (dates)
            # Days missing in either series become NaNs
            active_ret = returns - bench_series
            active_ret = active_ret.dropna() # Remove mismatched days

            if not active_ret.empty:
                # Tracking Error: Volatility of the difference between Portfolio and Benchmark
                te = active_ret.std() * np.sqrt(252)
                
                # Information Ratio: Active Return per unit of Active Risk (TE)
                mean_active_ret = active_ret.mean() * 252
                
                if te > 0.0001: 
                    ir = mean_active_ret / te
                    
        except Exception as e:
            print(f"[ERROR] TE Calc Failed: {e}")

    return {
        "Expected Return": round(mu, 4),
        "Volatility": round(sigma, 4),
        "Sharpe Ratio": round(sharpe, 4),
        "Sortino Ratio": round(sortino, 4),
        "Omega Ratio": round(omega, 4),
        "CVaR (95%)": round(abs(cvar_95), 4), 
        "Max Drawdown": round(max_dd, 4),
        "Tracking Error": round(te, 4),      
        "Information Ratio": round(ir, 4)    
    }

def get_risk_contribution(weights, cov_matrix):
    """
    Calculates percentage risk contribution of each asset.
    Used for Risk Parity charts.
    Formula: RC_i = w_i * (MCR_i) / Total_Vol
    """
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Marginal Contribution to Risk
    mcr = np.dot(cov_matrix, weights) / port_vol
    # Absolute Risk Contribution
    rc = weights * mcr
    # Percentage Risk Contribution
    rc_percent = rc / port_vol
    return rc_percent