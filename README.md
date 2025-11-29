# Kalpi Capital Portfolio Optimizer 📈

**A full-stack quantitative finance system for portfolio optimization, backtesting, and risk analysis.**

This project combines **FastAPI** (Backend Engine), **Streamlit** (Frontend Dashboard), and **Riskfolio-Lib** (Optimization Core) to create a powerful and interactive portfolio analytics toolkit.

---

## 🚀 Features

### Optimization Strategies
* **Mean-Variance:** Max Sharpe, Min Volatility.
* **Tail Risk:** CVaR (Conditional Value at Risk).
* **Alternative:** Risk Parity, Kelly Criterion.
* **Ratios:** Max Sortino, Max Omega.
* **Active Management:** Tracking Error & Information Ratio.

### Analytics & Constraints
* **Visualizations:** Efficient Frontier (with Feasible Cloud), Risk Contribution, Factor Exposure (Beta).
* **Constraints:** Long-only, Min/Max weights, L1 Sparsity (Clean Weights).
* **Backtesting:** Cumulative Returns, Drawdown Charts, and Monthly Heatmaps.

---

## 📸 Screenshots

### **1. Main Dashboard & Metrics**

Comprehensive portfolio analytics including Sharpe Ratio, CVaR, and active management metrics.
![Main Dashboard](screenshots/dashboard_main.png)

### **2. Efficient Frontier**

Visualizes the optimal risk/return tradeoff with a Monte Carlo simulation cloud.
![Efficient Frontier](screenshots/efficient_frontier.png)

### **3. Allocation & Risk Analysis**
Breakdown of optimal weights and factor exposure (Beta) against the benchmark.

![asset_allocation](screenshots/asset_allocation.png)
![factor_exposure](screenshots/factor_exposure.png) 


### 4. **Backtesting Engine**

Historical performance simulation including Growth, Drawdown analysis, and Monthly Return Heatmaps.
![Backtest Charts](screenshots/backtest_charts.png)
![Returns Heatmap](screenshots/returns_heatmap.png)

---

## 📂 Project Structure

```text
├── backend/
│   ├── main.py          # FastAPI Entry Point
│   ├── logic.py         # Core Optimization Logic
│   ├── data_loader.py   # Yahoo Finance Data Fetcher
│   └── metrics.py       # Financial Metrics
│
├── frontend/
│   └── app.py           # Streamlit Dashboard
│
├── screenshots/         # Project Images
│   ├── dashboard_main.png
│   ├── efficient_frontier.png
│   └── ...
│
├── requirements.txt     # Dependencies
├── Dockerfile           # Image Configuration
├── docker-compose.yml   # Container Orchestration
└── README.md
```
---

## Installation & Usage

**Setup Environment**

```bash
# Clone the repository
git clone [https://github.com/Aditya123500/kalpi-portfolio.git](https://github.com/Aditya123500/kalpi-portfolio.git)
cd kalpi-portfolio
```
### **Option A: Using Docker**

Prerequisite: Ensure Docker Desktop is installed and running.

Run the App: Open a terminal in the project root and run:

```bash
docker-compose up --build
```
Streamlit will show something like:
You can now view your Streamlit app in your browser.
```
Local URL: http://localhost:8501
Network URL: http://172.17.0.2:8501
```
Access the Dashboard: Open the Local URL and you'll navigate to: http://localhost:8501

Stop the App: Press Ctrl+C in the terminal to stop the containers.

---

### **Option B: Local Python Setup**

```bash
Prerequisite: 
             # Install Dependencies
               pip install -r requirements.txt
```
1. Run the Backend (Terminal 1)

```bash
python backend/main.py
```
# Server starts at [http://127.0.0.1:8000](http://127.0.0.1:8000)

2. Run the Frontend (Terminal 2) Open a new terminal window/tab and run:

```bash
streamlit run frontend/app.py
```
# Dashboard opens at http://localhost:8501

---

## 📡 API Example
The backend exposes a REST API for optimization. You can test it using Postman or cURL.

Endpoint: POST /optimize

Payload:
JSON

{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "method": "Mean-Variance - Max Sharpe",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "min_weight": 0.0,
  "max_weight": 1.0,
  "risk_free_rate": 0.02,
  "sparse": false
}

---

## 💻 Tech Stack
Python 3.9+
Riskfolio-Lib: Portfolio Optimization Engine
FastAPI: High-performance Backend API.
Streamlit: Interactive Frontend Dashboard.
Plotly: Interactive Financial Charts.
YFinance: Market Data Source.

---
