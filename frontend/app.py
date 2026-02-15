import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# Configure page layout and title
st.set_page_config(page_title="Kalpi Capital", page_icon="📈", layout="wide")
st.title("Kalpi Capital Portfolio Optimizer")

# --- 1. SIDEBAR: ASSETS & TIME INPUTS ---
st.sidebar.header("1. Assets & Time")
default_tickers = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, AMZN, MSFT"
tickers_in = st.sidebar.text_area("Tickers", default_tickers)
# Parse comma-separated tickers, strip whitespace, uppercase
tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]

# Benchmark for Alpha/Beta/TE calculation (Dynamic input)
bench_in = st.sidebar.text_input("Benchmark Ticker (e.g. SPY)", value="^NSEI")

today = datetime.date.today()
start = st.sidebar.date_input("Start", today - datetime.timedelta(days=730)) # Default 2 years back
end = st.sidebar.date_input("End", today)

# --- 2. SIDEBAR: STRATEGY SELECTION ---
st.sidebar.header("2. Methodology")
method = st.sidebar.selectbox("Optimizer", [
    "Mean-Variance - Max Sharpe",
    "Mean-Variance - Min Variance",
    "Min CVaR",
    "Risk Parity",
    "Tracking Error - Min Tracking Error",
    "Tracking Error - Max Info Ratio",
    "Kelly",
    "Max Sortino",
    "Max Omega",
    "Min Max Drawdown"
])

# --- 3. SIDEBAR: DYNAMIC PARAMETERS ---
# Show/Hide inputs based on selected strategy to keep UI clean
st.sidebar.header("3. Parameters")

# Defaults
rf = 0.0
mar = 0.0
alpha = 0.05

show_rf = False
show_mar = False
show_alpha = False
show_frontier_points = False

# Conditional Logic for parameter visibility
if "Mean-Variance" in method or "Max Info Ratio" in method:
    show_rf = True
    if "Mean-Variance" in method:
        show_frontier_points = True
elif "Max Sortino" in method:
    show_rf = True
    show_mar = True
elif "Max Omega" in method:
    show_rf = True
    show_mar = True
elif "CVaR" in method:
    show_rf = True
    show_alpha = True
elif "Min Max Drawdown" in method:
    show_rf = True

# Render inputs if flag is True
if show_rf:
    rf = st.sidebar.number_input("Risk Free Rate", 0.0, 0.2, 0.0, 0.01)
if show_mar:
    mar = st.sidebar.number_input("MAR (Min Acceptable Return)", 0.0, 0.5, 0.0, 0.01)
if show_alpha:
    alpha = st.sidebar.number_input("Confidence Level (CVaR)", 0.01, 0.5, 0.05, 0.01)

# Frontier resolution (Only for Mean-Variance)
n_points = 20
if show_frontier_points:
    n_points = st.sidebar.number_input("Number of points on efficient frontier (for MVO)", min_value=20, max_value=100, value=30, step=5)


# --- 4. SIDEBAR: CONSTRAINTS & COSTS ---
st.sidebar.header("4. Constraints & Costs")
sparse = st.sidebar.checkbox("Force L1 Sparsity (Clean Weights)", False) # Removes tiny positions
tx_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 2.0, 0.0, 0.1)

c1, c2 = st.sidebar.columns(2)
# Min Weight restricted to 0.0 (Long Only)
min_w = c1.number_input("Min Weight", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
max_w = c2.number_input("Max Weight", min_value=0.0, max_value=1.0, value=1.0, step=0.05)


# --- MAIN EXECUTION BLOCK ---

# Initialize Session State to hold data across re-runs
if "opt_data" not in st.session_state:
    st.session_state["opt_data"] = None
if "frontier_data" not in st.session_state:
    st.session_state["frontier_data"] = None

# BUTTON CLICK: Only fetches data, does not render
if st.sidebar.button("Run Optimization ⚡"):
    with st.spinner("Optimizing..."):
        try:
            # 1. Prepare Payload for API
            payload = {
                "tickers": tickers, 
                "benchmark": bench_in,
                "start_date": str(start), "end_date": str(end),
                "method": method, "risk_free_rate": rf,
                "mar": mar, "alpha": alpha, "sparse": sparse,
                "long_short": False, # Hardcoded: No Shorting
                "min_weight": min_w, "max_weight": max_w
            }
            
            # Default to localhost for local testing, but allow Docker to override it
            API_URL = os.getenv("API_URL", "https://kalpi-backend.onrender.com") 
            res = requests.post(f"{API_URL}/optimize", json=payload)
            
            if res.status_code == 200:
                st.session_state["opt_data"] = res.json()
                st.session_state["frontier_data"] = None # Reset frontier
                
                # If MVO, fetch frontier immediately
                if "Mean-Variance" in method:
                     f_res = requests.post(f"{API_URL}/frontier", json={
                        "tickers": tickers, 
                        "benchmark": bench_in, 
                        "start_date": str(start), "end_date": str(end),
                        "num_points": n_points,
                        "min_weight": min_w, "max_weight": max_w,
                        "risk_free_rate": rf 
                    })
                     if f_res.status_code == 200:
                         st.session_state["frontier_data"] = f_res.json()
                
                st.success("Optimization Complete!")
            else:
                st.error(f"Backend Error: {res.text}")

        except Exception as e:
            st.error(f"Connection Failed: {e}")

# RENDERING BLOCK: Checks if data exists in state
if st.session_state["opt_data"]:
    data = st.session_state["opt_data"]
    
    actual_method = data.get("method_used", method)
    st.success(f"Result: {actual_method}")
    
    # Show warnings if any assets were dropped
    if "messages" in data:
        for msg in data["messages"]: st.warning(msg)

    # --- VISUALIZATION SECTION ---

    # A. Performance Metrics Grid (3x4 Layout)
    st.subheader("Performance Metrics")
    m = data['metrics']
    net_return = m['Expected Return'] - (tx_cost/100)
    
    # Row 1: Basic Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exp. Return", f"{m['Expected Return']*100:.2f}%")
    c2.metric("Net Return", f"{net_return*100:.2f}%", delta=f"-{tx_cost}% Cost")
    c3.metric("Volatility", f"{m['Volatility']*100:.2f}%")
    c4.metric("Tracking Error", f"{m['Tracking Error']*100:.2f}%") 

    # Row 2: Risk Ratios
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sharpe", m['Sharpe Ratio'])
    c6.metric("Sortino", m['Sortino Ratio'])
    c7.metric("Omega", m['Omega Ratio'])
    c8.metric("Info Ratio", m['Information Ratio'])

    # Row 3: Tail Risk & Beta
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("CVaR (95%)", f"{m['CVaR (95%)']*100:.2f}%")
    c10.metric("Max DD", f"{m['Max Drawdown']*100:.2f}%")
    c11.metric("Beta (Mkt)", m['Beta'])
    c12.metric("Alpha", m['Alpha'])
    
    # B. Asset Allocation (Pie Chart)
    st.subheader("Asset Allocation")
    w_df = pd.DataFrame(list(data['weights'].items()), columns=["Asset", "Weight"])
    
    # Filter out zero weights for cleaner chart
    w_df = w_df[w_df["Weight"] > 0.0001]
    
    # Detect Cash/Unallocated if limits are strict (sum < 99%)
    total_weight = w_df["Weight"].sum()
    if total_weight < 0.99:
        cash_weight = 1.0 - total_weight
        new_row = pd.DataFrame([{"Asset": "CASH / UNALLOCATED", "Weight": cash_weight}])
        w_df_chart = pd.concat([w_df, new_row], ignore_index=True)
    else:
        w_df_chart = w_df

    c_pie, c_dl = st.columns([3, 1])
    with c_pie:
        fig_alloc = px.pie(w_df_chart, values="Weight", names="Asset", hole=0.4)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with c_dl:
        st.write("###")
        csv = w_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "weights.csv", "text/csv")
    
    # C. Factor Exposure (Beta Bar Chart)
    if "factor_exposure" in data and data["factor_exposure"]:
        st.subheader("Factor Exposure (Market Beta)")
        
        beta_df = pd.DataFrame(list(data['factor_exposure'].items()), columns=["Asset", "Beta"])
        port_beta = m['Beta']
        
        fig_beta = px.bar(
            beta_df, 
            x="Asset", y="Beta", 
            text_auto='.2f',
            title=f"Asset Sensitivity vs Benchmark (Portfolio Beta: {port_beta})",
            color="Beta",
            color_continuous_scale="RdBu_r"
        )
        # Add Market Reference Line (1.0)
        fig_beta.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="Market (1.0)")
        st.plotly_chart(fig_beta, use_container_width=True)

    # D. Risk Contribution (Risk Parity Only)
    if "Risk Parity" in method:
        st.subheader("Risk Contribution (Target: Equal Risk)")
        rc_df = pd.DataFrame(list(data['risk_contribution'].items()), columns=["Asset", "Risk Contribution"])
        
        fig_rc = px.bar(
            rc_df, 
            x="Asset", y="Risk Contribution",
            color="Risk Contribution",
            color_continuous_scale="Viridis",
            text_auto='.1%'
        )
        fig_rc.update_layout(yaxis_tickformat=".1%", showlegend=False)
        st.plotly_chart(fig_rc, use_container_width=True)

    # E. Growth & Drawdown Charts
    c_growth, c_dd = st.columns(2)
    with c_growth:
        st.subheader("📈 Backtest Growth ($100)")
        gh_df = pd.DataFrame(list(data['growth_history'].items()), columns=["Date", "Value"])
        st.plotly_chart(px.line(gh_df, x="Date", y="Value"), use_container_width=True)
    
    with c_dd:
        st.subheader("📉 Drawdown Chart")
        dd_df = pd.DataFrame(list(data['drawdown_history'].items()), columns=["Date", "Drawdown"])
        st.plotly_chart(px.area(dd_df, x="Date", y="Drawdown", color_discrete_sequence=['red']), use_container_width=True)

    # F. Heatmap
    st.subheader("Monthly Returns Heatmap")
    if "heatmap_data" in data:
        heat_df = pd.DataFrame.from_dict(data["heatmap_data"], orient='index')
        heat_df.index = heat_df.index.astype(str)
        # Sort columns chronologically
        cols_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        valid_cols = [c for c in cols_order if c in heat_df.columns]
        heat_df = heat_df.reindex(columns=valid_cols)
        
        fig_heat = px.imshow(heat_df, text_auto=".1%", aspect="auto", color_continuous_scale="RdBu_r", origin='lower')
        st.plotly_chart(fig_heat, use_container_width=True)

    # 3. Efficient Frontier Rendering (From State)
    if st.session_state["frontier_data"]:
        data_f = st.session_state["frontier_data"]
        df_f = pd.DataFrame(data_f['frontier'])
        df_r = pd.DataFrame(data_f['random'])
        
        st.subheader("Efficient Frontier")
        fig_f = go.Figure()

        # Layer 1: Random Cloud (Feasible Region)
        if not df_r.empty:
            fig_f.add_trace(go.Scatter(
                x=df_r["Volatility"], y=df_r["Return"], 
                mode='markers', name='Feasible Portfolios',
                marker=dict(color='lightgrey', size=4, opacity=0.5)
            ))

        # Layer 2: Frontier Line (Optimal Curve)
        if not df_f.empty:
            fig_f.add_trace(go.Scatter(
                x=df_f["Volatility"], y=df_f["Return"], 
                mode='lines+markers', name='Efficient Frontier',
                marker=dict(size=8, color=df_f["Sharpe"], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
                line=dict(color='black', width=1, shape='spline') # Spline smooths the curve
            ))

        # Layer 3: Selected Portfolio (Red Star)
        if 'm' in locals():
            fig_f.add_trace(go.Scatter(
                x=[m['Volatility']], y=[m['Expected Return']], 
                mode='markers', marker=dict(size=18, color='red', symbol='star'), 
                name="Selected Portfolio"
            ))

        fig_f.update_layout(xaxis_title="Risk (Volatility)", yaxis_title="Expected Return", height=600)
        st.plotly_chart(fig_f, use_container_width=True)
