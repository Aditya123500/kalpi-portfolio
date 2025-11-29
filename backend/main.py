from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
from data_loader import fetch_data
# Import portfolio logic and helper functions
from logic import PortfolioOptimizer, get_efficient_frontier, get_random_portfolios

# Initialize FastAPI app with metadata
app = FastAPI(title="Kalpi Capital Engine")

# --- DATA MODELS (Pydantic) ---
# Defines the expected structure of incoming JSON requests

class DataRequest(BaseModel):
    """Simple request for fetching raw data (unused in current endpoints but good for testing)."""
    tickers: List[str]
    start_date: str
    end_date: str

class FrontierRequest(BaseModel):
    """
    Request model for the Efficient Frontier chart.
    Requires constraints (min/max weight, rf) to ensure the chart matches the optimized portfolio.
    """
    tickers: List[str]
    benchmark: Optional[str] = "SPY" # Default to SPY if not provided
    start_date: str
    end_date: str
    num_points: int = 30 # Resolution of the frontier curve (dots count)
    # Constraints passed to ensure the frontier line respects user rules
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_free_rate: float = 0.0

class OptimizationRequest(BaseModel):
    """
    Request model for the main Single-Period Optimization.
    Contains all strategy parameters, constraints, and backtest options (if extended).
    """
    tickers: List[str]
    benchmark: Optional[str] = "^NSEI" # Default to Nifty 50
    start_date: str
    end_date: str
    method: str # Strategy name (e.g. "Max Sharpe")
    risk_free_rate: float = 0.0
    mar: float = 0.0 # Min Acceptable Return (for Sortino/Omega)
    alpha: float = 0.05 # Confidence level for CVaR
    sparse: bool = False # L1 Sparsity flag (clean tiny weights)
    long_short: bool = False # (Currently unused/hardcoded false in logic)
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    
# --- API ENDPOINTS ---

@app.post("/optimize")
def optimize_portfolio(req: OptimizationRequest):
    """
    Main endpoint: Calculates optimal weights and performance metrics for a single period.
    """
    try:
        # 1. Fetch Historical Price Data
        prices = fetch_data(req.tickers, req.start_date, req.end_date)
        
        # 2. Fetch Benchmark Data (Optional)
        bench_prices = None
        if req.benchmark:
            try:
                bench_prices = fetch_data([req.benchmark], req.start_date, req.end_date)
            except: pass # Continue even if benchmark fails (metrics will just skip Alpha/Beta)

        # 3. Initialize Optimization Engine
        optimizer = PortfolioOptimizer(prices, benchmark_prices=bench_prices)
        
        # 4. Package User Constraints
        constraints = {
            "risk_free_rate": req.risk_free_rate,
            "mar": req.mar,
            "alpha": req.alpha,
            "sparse": req.sparse,
            "long_short": req.long_short,
            "min_weight": req.min_weight,
            "max_weight": req.max_weight
        }
        
        # 5. Run Optimization Strategy
        result = optimizer.optimize(req.method, constraints)
        return result

    except Exception as e:
        # Return 400 Error with message if something breaks (e.g. No data found)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/frontier")
def frontier_endpoint(req: FrontierRequest):
    """
    Endpoint for Efficient Frontier Visualization.
    Returns two datasets:
    1. 'frontier': Points forming the optimal curve.
    2. 'random': Random portfolios forming the feasible cloud.
    """
    try:
        # 1. Fetch Data
        prices = fetch_data(req.tickers, req.start_date, req.end_date)
        
        # 2. Fetch Benchmark (Needed to align dates exactly like the Optimizer)
        bench_prices = None
        if req.benchmark:
            try:
                bench_prices = fetch_data([req.benchmark], req.start_date, req.end_date)
            except: pass
        
        # 3. Setup Constraints (Must match what user selected for consistency)
        constraints = {
            "min_weight": req.min_weight,
            "max_weight": req.max_weight,
            "long_short": False
        }

        # 4. Generate The Curve (Efficient Frontier)
        frontier_points = get_efficient_frontier(
            prices, 
            benchmark_prices=bench_prices, # Passed for date alignment
            points=req.num_points, 
            constraints=constraints, 
            rf=req.risk_free_rate
        )
        
        # 5. Generate The Cloud (Random Portfolios)
        random_points = get_random_portfolios(
            prices, 
            benchmark_prices=bench_prices, # Passed for date alignment
            constraints=constraints
        )
        
        return {
            "frontier": frontier_points, 
            "random": random_points
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
      

if __name__ == "__main__":
    # Start the Uvicorn server locally
    uvicorn.run(app, host="127.0.0.1", port=8000)