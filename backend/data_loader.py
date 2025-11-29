import yfinance as yf
import pandas as pd
from typing import List

def fetch_single_ticker(ticker: str, start: str, end: str):
    """
    Helper function: Tries to download a single ticker.
    1. Tries exactly what the user typed.
    2. If that fails, tries adding .NS (for Indian stocks).
    """
    ticker = ticker.strip().upper() # Normalize input
    
    # Attempt 1: Try downloading exact ticker symbol
    print(f"[INFO] Checking {ticker}...")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    
    # Check if we got data
    if not data.empty:
        return data, ticker
    
    # Attempt 2: If failed, try appending .NS (NSE India) automatically
    if "." not in ticker and "-" not in ticker:
        ns_ticker = f"{ticker}.NS"
        print(f"[INFO] {ticker} failed. Retrying as {ns_ticker}...")
        data = yf.download(ns_ticker, start=start, end=end, progress=False, auto_adjust=True)
        
        if not data.empty:
            return data, ns_ticker

    return None, None

def fetch_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Iterates through list of tickers and finds valid data for each.
    """
    combined_data = pd.DataFrame()
    valid_tickers = []

    for t in tickers:
        df, valid_name = fetch_single_ticker(t, start_date, end_date)
        
        if df is not None:
            # Handle yfinance's variable column structure (MultiIndex vs Flat)
            if isinstance(df.columns, pd.MultiIndex):
                # Handle yfinance multi-index (Price, Ticker)
                try:
                    # Priority: Close > Adj Close > First Column
                    if 'Close' in df.columns:
                        series = df['Close']
                    elif 'Adj Close' in df.columns:
                        series = df['Adj Close']
                    else:
                        series = df.iloc[:, 0]
                except:
                    series = df.iloc[:, 0]
            else:
                # Handle Flat dataframe
                col = 'Close' if 'Close' in df.columns else df.columns[0]
                series = df[col]

            # Rename series to the valid ticker name
            series.name = valid_name
            
            # Merge into main dataframe (Outer join preserves disparate dates initially)
            if combined_data.empty:
                combined_data = pd.DataFrame(series)
            else:
                combined_data = combined_data.join(series, how='outer')
            
            valid_tickers.append(valid_name)
        else:
            print(f"[WARN] Could not find data for '{t}'. Skipping.")

    # Final Cleaning
    if combined_data.empty:
        raise ValueError("No valid data found for any of the provided tickers.")

    # 1. Remove rows where market was closed everywhere (Global holidays)
    combined_data.dropna(how='all', inplace=True)
    # 2. Forward fill gaps (Fixes mismatch if one market is open while another is closed)
    combined_data.ffill(inplace=True)
    # 3. Trim start dates so all assets align perfectly
    combined_data.dropna(inplace=True)

    print(f"[SUCCESS] optimized data for: {valid_tickers}")
    return combined_data