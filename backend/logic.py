import riskfolio as rp
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from metrics import calculate_portfolio_metrics, get_risk_contribution

warnings.filterwarnings("ignore")

# --- BASE OPTIMIZER ---
class BaseOptimizer:
    """
    Parent class handling data alignment, constraint setup, and result cleaning.
    Shared by all specific strategy optimizers.
    """
    def __init__(self, prices: pd.DataFrame, benchmark_prices: pd.DataFrame = None):
        if len(prices.columns) < 2:
            raise ValueError(f"Need at least 2 stocks! Provided: {len(prices.columns)}")
        
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        
        # FIX: Remove Timezone info to ensure date matching works across different data sources
        if self.returns.index.tz is not None:
            self.returns.index = self.returns.index.tz_localize(None)
        
        # N+5 Rule: Ensure we have enough data points (days) vs assets to calculate covariance
        if len(self.returns) < len(prices.columns) + 5:
            raise ValueError("Date range too short (N+5 rule). Increase range.")

        self.bench_returns = None
        if benchmark_prices is not None:
            self.bench_returns = benchmark_prices.pct_change().dropna()
            
            # FIX: Ensure Benchmark is also Timezone Naive
            if self.bench_returns.index.tz is not None:
                self.bench_returns.index = self.bench_returns.index.tz_localize(None)

            # Align Dates: Keep only rows present in BOTH Portfolio and Benchmark
            common = self.returns.index.intersection(self.bench_returns.index)
            
            if len(common) < 10:
                print("[WARN] Benchmark and Portfolio have almost no overlapping dates! Metrics will be 0.")
                self.bench_returns = None # Discard benchmark if alignment fails
            else:
                self.returns = self.returns.loc[common]
                self.bench_returns = self.bench_returns.loc[common]

        # Initialize Riskfolio Portfolio Object
        self.port = rp.Portfolio(returns=self.returns)
        self.port.assets_stats(method_mu='hist', method_cov='hist')
        
        # Constraint placeholders
        self.user_max_weight = 1.0
        self.solver_max_weight = 1.0
        self.active_min_weight = 0.0

    def setup_constraints(self, constraints):
        """
        Parses user constraints (Min/Max Weight) and applies them to the optimizer.
        Handles the edge case where constraints are mathematically impossible.
        """
        if constraints is None: return

        raw_min = float(constraints.get('min_weight', 0.0))
        raw_max = float(constraints.get('max_weight', 1.0))
        
        # Scale percentages (e.g. 50 -> 0.5)
        if raw_max > 1.0: raw_max /= 100.0
        if raw_min > 1.0: raw_min /= 100.0
        if raw_min < 0.0: raw_min = 0.0 # Force Long-Only

        n_assets = len(self.prices.columns)
        
        # Store user's strict desire for later clipping
        self.user_max_weight = raw_max
        self.active_min_weight = raw_min

        # SOLVER FIX: If Max Weight is too low (e.g. 15% for 7 stocks = 105% required, but user gave 15%),
        # the solver will crash. We calculate the minimum feasible max weight (1/N) and use that
        # if the user's input is too strict. We then clip the results back down later.
        min_feasible_max = 1.0 / n_assets
        if raw_max < min_feasible_max:
            self.solver_max_weight = min_feasible_max + 0.001 # Add buffer
        else:
            self.solver_max_weight = raw_max

        # Apply relaxed limits to Riskfolio engine
        self.port.lb = pd.Series(raw_min, index=self.prices.columns)
        self.port.ub = pd.Series(self.solver_max_weight, index=self.prices.columns)

    def _enforce_hard_limits(self, weights, sparse=False):
        if weights is None: return None
        
        w_arr = weights.to_numpy().flatten()
        strict_max = self.user_max_weight
        strict_min = self.active_min_weight 
        
        # 1. Sparsity (Clean dust)
        threshold = 0.015 if sparse else 0.00001
        w_arr[np.abs(w_arr) < threshold] = 0.0

        # 2. Normalize to 100% (The main scaling step)
        current_sum = np.sum(w_arr)
        if abs(current_sum) > 0.0001:
            w_arr = w_arr / current_sum

        # 3. Apply Max Cap
        w_arr = np.minimum(w_arr, strict_max)

        # 4. Apply Min Floor (THE FINAL STEP)
        # We assume that if Sum > 100% after this, it's better to have 100.1% 
        # than to violate the user's hard constraint.
        active_mask = w_arr > 0.0001
        w_arr[active_mask] = np.maximum(w_arr[active_mask], strict_min)

        # DO NOT NORMALIZE AGAIN HERE. 
        # Returning sum=1.002 is safer than returning weight=5.9% when min=6%.

        return pd.DataFrame(data=w_arr, index=weights.index, columns=['weights'])

    def _clean_output(self, weights, method_name, rf=0.0, mar=0.0, sparse=False, tx_cost=0.0):
        """
        Standardizes the output format: Weights, Metrics, Charts.
        """
        # Apply strict limits & sparsity
        weights = self._enforce_hard_limits(weights, sparse=sparse)

        if weights is None:
            weights = self._fallback_weights()
            method_name = f"{method_name} (FAILED -> USED EQUAL WEIGHTS)"
        
        w_series = weights.iloc[:, 0]
        port_daily_ret = self.returns.dot(w_series)
        
        # Prepare Benchmark for Metrics
        bench_series = None
        if self.bench_returns is not None:
            bench_series = self.bench_returns.iloc[:, 0]

        # Calculate all metrics (Sharpe, TE, IR, etc.)
        metrics = calculate_portfolio_metrics(
            port_daily_ret, 
            benchmark_returns=bench_series,
            risk_free_rate=rf, 
            mar=mar
        )
        
        # Adjust for Transaction Costs
        adjusted_mu = round(metrics["Expected Return"] - (tx_cost / 100), 4)
        metrics["Expected Return"] = adjusted_mu
        
        # Recalculate Sharpe with new return
        if metrics["Volatility"] > 0:
            metrics["Sharpe Ratio"] = round((adjusted_mu - rf) / metrics["Volatility"], 4)

        # Calculate Beta/Alpha vs Benchmark
        beta, alpha = 0.0, 0.0
        if self.bench_returns is not None:
            aligned_port = port_daily_ret.loc[self.bench_returns.index]
            aligned_bench = self.bench_returns.iloc[:, 0]
            if len(aligned_port) > 0:
                s, i, _, _, _ = stats.linregress(aligned_bench, aligned_port)
                beta, alpha = s, i * 252 
        metrics["Beta"] = round(beta, 2)
        metrics["Alpha"] = round(alpha, 4)

        # Generate Chart Data
        cov = self.returns.cov() * 252
        risk_contrib = get_risk_contribution(w_series.values, cov.values)
        risk_contrib_dict = dict(zip(self.prices.columns, np.round(risk_contrib, 4)))

        cum_ret = (1 + port_daily_ret).cumprod() * 100
        growth_hist = {str(k.date()): v for k, v in cum_ret.items()}

        running_max = (1 + port_daily_ret).cumprod().cummax()
        drawdown = ((1 + port_daily_ret).cumprod() / running_max) - 1
        drawdown_hist = {str(k.date()): v for k, v in drawdown.items()}
        
        rolling_sharpe = (port_daily_ret.rolling(126).mean() / port_daily_ret.rolling(126).std()) * np.sqrt(252)
        rolling_sharpe_hist = {str(k.date()): v for k, v in rolling_sharpe.dropna().items()}

        annual_ret = port_daily_ret.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        annual_history = {str(k.year): v for k, v in annual_ret.items()}

        monthly_ret = port_daily_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        heatmap_data = {}
        for date, val in monthly_ret.items():
            year = str(date.year)
            month = date.strftime('%b')
            if year not in heatmap_data: heatmap_data[year] = {}
            heatmap_data[year][month] = round(val, 4)
        
        factor_exposure = self._calculate_factor_exposure()    

        return {
            "weights": w_series.to_dict(),
            "metrics": metrics,
            "risk_contribution": risk_contrib_dict,
            "factor_exposure": factor_exposure,
            "growth_history": growth_hist,
            "drawdown_history": drawdown_hist,
            "rolling_sharpe_history": rolling_sharpe_hist,
            "annual_history": annual_history,
            "heatmap_data": heatmap_data,
            "method_used": method_name
        }

    def _fallback_weights(self):
        # Used if solver fails: Returns Equal Weights clipped to User Max
        n = len(self.prices.columns)
        w = min(1.0/n, self.user_max_weight)
        return pd.DataFrame([w]*n, index=self.prices.columns, columns=['weights'])

    def _solve_standard(self, **kwargs):
        # Wrapper to try default solver, then fallback
        try:
            if 'solver' not in kwargs: kwargs['solver'] = 'SCS'
            return self.port.optimization(**kwargs)
        except:
            try:
                if 'solver' in kwargs: del kwargs['solver']
                return self.port.optimization(**kwargs)
            except:
                return None

    def _solve_monte_carlo(self, method, iterations=5000, rf=0.0, mar=0.0):
        # Used for non-convex problems like Kelly / Omega
        n_assets = len(self.prices.columns)
        best_score = -np.inf
        best_weights = None
        mean_ret = self.returns.mean() * 252
        cov = self.returns.cov() * 252
        
        min_w = self.active_min_weight 
        max_w = self.user_max_weight 
        range_span = max_w - min_w
        if min_w > max_w: min_w = max_w

        for _ in range(iterations):
            w = np.random.random(n_assets)
            w = min_w + (w * range_span)
            
            if abs(np.sum(w)) > 0.01: w = w / np.sum(w)
            w = np.clip(w, min_w, max_w)
            
            ret = np.sum(mean_ret * w)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            
            if method == "Kelly": score = ret - (vol**2)/2
            elif method == "Max Omega": score = (ret - mar)/vol if vol > 0 else 0
            else: score = 0
            
            if score > best_score: best_score, best_weights = score, w
            
        if best_weights is None: return self._fallback_weights()
        return pd.DataFrame(data=best_weights, index=self.prices.columns, columns=['weights'])
    
    def _calculate_factor_exposure(self):
        """
        Calculates simple CAPM Beta for each asset individually against the benchmark.
        Used for the 'Factor Exposure' chart.
        """
        exposure = {}
        if self.bench_returns is None or self.bench_returns.empty:
            return {}

        market_ret = self.bench_returns.iloc[:, 0]

        for asset in self.returns.columns:
            asset_ret = self.returns[asset]
            
            common = asset_ret.index.intersection(market_ret.index)
            if len(common) > 10:
                y = asset_ret.loc[common]
                x = market_ret.loc[common]
                slope, _, _, _, _ = stats.linregress(x, y)
                exposure[asset] = round(slope, 2)
            else:
                exposure[asset] = 0.0
        
        return exposure


# --- STRATEGY CLASSES ---
# Each class configures Riskfolio differently based on the strategy goal.

class MeanVarianceOptimizer(BaseOptimizer):
    def optimize(self, goal, rf=0.0, sparse=False, tx=0.0):
        obj = 'Sharpe' if goal == 'Sharpe' else 'MinRisk'
        # Try Sharpe calc with RF
        weights = self._solve_standard(model='Classic', rm='MV', obj=obj, rf=rf)
        
        # If Sharpe fails (e.g. negative returns), retry with RF=0
        if weights is None and goal == 'Sharpe' and rf > 0:
            weights = self._solve_standard(model='Classic', rm='MV', obj='Sharpe', rf=0.0)
        
        # Fallback to Min Volatility if Sharpe fails entirely
        if weights is None:
            weights = self._solve_standard(model='Classic', rm='MV', obj='MinRisk', rf=rf)
            return self._clean_output(weights, "Mean-Variance - Min Variance (Fallback)", rf=rf, sparse=sparse, tx_cost=tx)
        
        return self._clean_output(weights, f"Mean-Variance - {goal}", rf=rf, sparse=sparse, tx_cost=tx)

class CvarOptimizer(BaseOptimizer):
    def optimize(self, alpha=0.05, rf=0.0, sparse=False, tx=0.0):
        # CVaR (Conditional Value at Risk) optimization
        weights = self._solve_standard(model='Classic', rm='CVaR', obj='MinRisk', rf=rf, alpha=alpha)
        if weights is None:
            # Fallback to Mean-Variance if CVaR solver fails
            weights = self._solve_standard(model='Classic', rm='MV', obj='MinRisk', rf=rf)
            return self._clean_output(weights, "Min Variance (CVaR Failed)", rf=rf, sparse=sparse, tx_cost=tx)
        return self._clean_output(weights, "Min CVaR", rf=rf, sparse=sparse, tx_cost=tx)

class RiskParityOptimizer(BaseOptimizer):
    def optimize(self, sparse=False, tx=0.0):
        try:
            # Risk Parity = Equal Risk Contribution
            weights = self.port.rp_optimization(model='Classic', rm='MV')
        except:
            weights = None
        return self._clean_output(weights, "Risk Parity", sparse=sparse, tx_cost=tx)

class TrackingErrorOptimizer(BaseOptimizer):
    def __init__(self, prices, benchmark_prices):
        super().__init__(prices, benchmark_prices)
        if self.bench_returns is None: raise ValueError("Benchmark required")
        # Calculate Active Returns (Asset - Benchmark)
        self.active_returns = self.returns.sub(self.bench_returns.iloc[:,0], axis=0)
        self.port_active = rp.Portfolio(returns=self.active_returns)
        self.port_active.assets_stats(method_mu='hist', method_cov='hist')

    def optimize(self, goal='MinTE', rf=0.0, sparse=False, tx=0.0):
        try:
            obj = 'MinRisk' if goal == 'MinTE' else 'Sharpe'
            name = "Min Tracking Error" if goal == 'MinTE' else "Max Info Ratio"
            weights = self.port_active.optimization(model='Classic', rm='MV', obj=obj, rf=rf)
            return self._clean_output(weights, name, rf=rf, sparse=sparse, tx_cost=tx)
        except:
            weights = self.port.optimization(model='Classic', rm='MV', obj='MinRisk', rf=rf)
            return self._clean_output(weights, "Min Variance (TE Failed)", rf=rf, sparse=sparse, tx_cost=tx)

class KellyOptimizer(BaseOptimizer):
    def optimize(self, rf=0.0, sparse=False, tx=0.0):
        # Uses Monte Carlo because Kelly criterion is non-convex
        weights = self._solve_monte_carlo("Kelly", rf=rf)
        return self._clean_output(weights, "Kelly Criterion", rf=rf, sparse=sparse, tx_cost=tx)

class RatioOptimizer(BaseOptimizer):
    def optimize(self, method, mar=0.0, rf=0.0, sparse=False, tx=0.0):
        weights = None
        if method == "Sortino":
            # CDaR is the optimization equivalent for Sortino (Downside Risk)
            weights = self._solve_standard(model='Classic', rm='CDaR', obj='Sharpe', rf=mar) 
        elif method == "Omega":
            weights = self._solve_monte_carlo("Max Omega", rf=rf, mar=mar)
        elif method == "MDD":
            # Max Drawdown optimization
            weights = self._solve_standard(model='Classic', rm='MDD', obj='MinRisk', rf=rf)
            
        if weights is None and method == "Sortino" and mar > 0:
             weights = self._solve_standard(model='Classic', rm='CDaR', obj='Sharpe', rf=0.0)

        if weights is None:
            weights = self._solve_standard(model='Classic', rm='MV', obj='MinRisk', rf=rf)
            return self._clean_output(weights, f"Min Variance ({method} Failed)", rf=rf, sparse=sparse, tx_cost=tx)

        return self._clean_output(weights, f"Max {method}", rf=rf, mar=mar, sparse=sparse, tx_cost=tx)

# --- HELPER FUNCTIONS ---
def get_random_portfolios(prices, benchmark_prices=None, constraints=None, num_portfolios=2000):
    """
    Generates the 'Cloud' of feasible portfolios for the Efficient Frontier chart.
    Uses Dirichlet distribution + Random Mix to ensure full coverage of the space.
    """
    # Initialize with Benchmark to ensure same data alignment
    opt = BaseOptimizer(prices, benchmark_prices=benchmark_prices)
    
    if constraints:
        opt.setup_constraints(constraints)
    
    min_w = opt.active_min_weight
    max_w = opt.user_max_weight
    
    mean_ret = opt.returns.mean() * 252
    cov = opt.returns.cov() * 252
    n_assets = len(prices.columns)
    
    # Scale simulations based on number of assets
    dynamic_num = max(2000, n_assets * 1000)
    num_portfolios = min(dynamic_num, 10000) 

    results = []
    risk_free_rate = 0.0 
    if constraints:
        risk_free_rate = constraints.get('risk_free_rate', 0.0)

    for _ in range(num_portfolios):
        # Use mixture of Dirichlet (Edges) and Random (Center)
        if np.random.random() > 0.5:
            w = np.random.dirichlet(np.ones(n_assets), size=1)[0]
        else:
            w = np.random.random(n_assets)
            w /= np.sum(w)

        # Apply limits
        range_span = max_w - min_w
        w = min_w + (w * range_span)
        
        if np.sum(w) > 0:
            w = w / np.sum(w)
        
        ret = np.sum(mean_ret * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        
        results.append({
            "Volatility": round(vol, 4),
            "Return": round(ret, 4),
            "Sharpe": round(sharpe, 4)
        })
        
    return results

def get_efficient_frontier(prices, benchmark_prices=None, points=30, constraints=None, rf=0.0):
    """
    Calculates the 'Line' (Efficient Frontier) using the optimizer.
    Accepts RF to ensuring matching math with the selected portfolio.
    """
    opt = BaseOptimizer(prices, benchmark_prices=benchmark_prices)
    if constraints:
        opt.setup_constraints(constraints)
    try:
        ws = opt.port.efficient_frontier(model='Classic', rm='MV', points=points, rf=rf, hist=True)
        frontier_data = []
        for col in ws.columns:
            w = ws[col].values
            ret = np.sum(opt.returns.mean() * w) * 252
            vol = np.sqrt(np.dot(w.T, np.dot(opt.returns.cov() * 252, w)))
            sharpe = (ret - rf) / vol if vol > 0 else 0
            frontier_data.append({
                "Volatility": round(vol, 4), 
                "Return": round(ret, 4), 
                "Sharpe": round(sharpe, 4)
            })
        return frontier_data
    except Exception as e:
        print(f"[ERROR] Frontier generation failed: {e}")
        return []
    
    
   # --- ROUTER CLASS ---
class PortfolioOptimizer:
    def __init__(self, prices, benchmark_prices=None):
        self.prices = prices
        self.bench = benchmark_prices

    def optimize(self, method, constraints):
        rf = constraints.get('risk_free_rate', 0.0)
        mar = constraints.get('mar', 0.0)
        alpha = constraints.get('alpha', 0.05)
        sparse = constraints.get('sparse', False)
        tx = constraints.get('tx_cost', 0.0)

        method_norm = method.replace('–', '-').lower().strip()
        print(f"[DEBUG] Processing Strategy: {method} (Normalized: {method_norm})")

        opt = None
        
        if "mean-variance" in method_norm or "mvo" in method_norm:
            opt = MeanVarianceOptimizer(self.prices, self.bench)
            opt.setup_constraints(constraints)
            goal = 'MinRisk' if 'min' in method_norm and 'variance' in method_norm else 'Sharpe'
            return opt.optimize(goal, rf, sparse, tx)
        
        elif "cvar" in method_norm:
            opt = CvarOptimizer(self.prices, self.bench)
            opt.setup_constraints(constraints)
            return opt.optimize(alpha, rf, sparse, tx)
            
        elif "risk parity" in method_norm:
            opt = RiskParityOptimizer(self.prices, self.bench)
            opt.setup_constraints(constraints) 
            return opt.optimize(sparse, tx)
            
        elif "kelly" in method_norm:
            opt = KellyOptimizer(self.prices, self.bench)
            opt.setup_constraints(constraints) 
            return opt.optimize(rf, sparse, tx)
            
        elif "tracking error" in method_norm:
            opt = TrackingErrorOptimizer(self.prices, self.bench)
            opt.setup_constraints(constraints)
            goal = 'MinTE' if 'min' in method_norm else 'MaxIR'
            return opt.optimize(goal, rf, sparse, tx)
            
        else:
            opt = RatioOptimizer(self.prices, self.bench)
            opt.setup_constraints(constraints)
            
            key = "Sortino"
            if "omega" in method_norm: key = "Omega"
            elif "drawdown" in method_norm: key = "MDD"
            
            return opt.optimize(key, mar, rf, sparse, tx) 
    