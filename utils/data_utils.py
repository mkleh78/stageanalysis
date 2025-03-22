import pandas as pd
import numpy as np
import yfinance as yf
import logging
import traceback
import os

logger = logging.getLogger('WeinsteinAnalyzer')

def normalize_ticker(ticker):
    """Normalize ticker symbols to be compatible with yfinance"""
    # Remove any whitespace
    ticker = ticker.strip()
    
    # Convert to uppercase
    ticker = ticker.upper()
    
    # Handle special cases for different exchanges
    # For European tickers that might use '.' instead of ',' for decimal
    ticker = ticker.replace(',', '.')
    
    # For indices, ensure proper prefix
    if ticker.startswith('^'):
        return ticker
    
    # For common indices, add the ^ prefix if missing
    common_indices = {
        'SPX': '^GSPC',  # S&P 500
        'DJI': '^DJI',   # Dow Jones
        'IXIC': '^IXIC', # NASDAQ
        'RUT': '^RUT',   # Russell 2000
        'GSPC': '^GSPC', # S&P 500
        'NDX': '^NDX',   # NASDAQ-100
        'VIX': '^VIX'    # Volatility Index
    }
    
    if ticker in common_indices:
        return common_indices[ticker]
    
    return ticker

def is_index(ticker, ticker_info=None):
    """Check if the ticker is an index to avoid self-comparison in market context"""
    # Common indices usually start with ^
    if ticker.startswith('^'):
        return True
    
    # Check if it's one of the known indices
    known_indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^NDX', '^VIX']
    if ticker in known_indices:
        return True
        
    # If we have ticker info, check if it's an index
    if ticker_info and 'market' in ticker_info and 'index' in ticker_info['market'].lower():
        return True
        
    return False

def get_safe_series(df, column):
    """
    Safely extract a series from a dataframe, handling various data formats.
    
    Args:
        df (pandas.DataFrame): The dataframe to extract from
        column (str): The column name to extract
        
    Returns:
        pandas.Series: The extracted series with proper numeric values
    """
    if column not in df.columns:
        return pd.Series(index=df.index)
        
    series = df[column]
    
    # Convert to a standard series if it's a DataFrame
    if isinstance(series, pd.DataFrame):
        try:
            series = series.iloc[:, 0]
        except:
            return pd.Series(index=df.index)
    
    # Ensure all values are numeric
    return pd.to_numeric(series, errors='coerce')

def download_ticker_data(ticker, period="1y", interval="1wk", max_retries=3):
    """
    Download ticker data from Yahoo Finance with retry mechanism
    
    Args:
        ticker (str): The ticker symbol
        period (str): The time period to download
        interval (str): The data interval
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        tuple: (data, warnings, errors, actual_period, actual_interval)
    """
    warnings = []
    errors = []
    
    # Normalize ticker
    normalized_ticker = normalize_ticker(ticker)
    
    # Track the actual period and interval used
    actual_period = period
    actual_interval = interval
    
    # Try to download data with retry
    for attempt in range(max_retries):
        try:
            data = yf.download(
                normalized_ticker, 
                period=actual_period, 
                interval=actual_interval, 
                progress=False
            )
            
            if len(data) > 0:
                break
            elif attempt < max_retries - 1:
                # Try with a shorter period if no data is returned
                if actual_period == "5y":
                    actual_period = "2y"
                elif actual_period == "2y":
                    actual_period = "1y"
                elif actual_period == "1y":
                    actual_period = "6mo"
                elif actual_period == "6mo":
                    actual_period = "3mo"
                elif actual_period == "3mo":
                    actual_period = "1mo"
                else:
                    # If we're already at the shortest period, try a different interval
                    if actual_interval == "1wk":
                        actual_interval = "1d"
                        actual_period = "1mo"  # Reset period for daily data
                    
                logger.warning(f"Attempt {attempt+1}: No data returned for {ticker}. Trying with period={actual_period}, interval={actual_interval}")
            else:
                logger.error(f"No data found for {ticker} after {max_retries} attempts")
                errors.append(f"No data found for {ticker}. The ticker may not exist or may not have data for the requested period.")
                return None, warnings, errors, actual_period, actual_interval
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
            else:
                errors.append(f"Failed to download data for {ticker}: {str(e)}")
                return None, warnings, errors, actual_period, actual_interval
    
    # Check if we have enough data
    if len(data) < 5:  # Need at least 5 data points for minimal analysis
        logger.warning(f"Insufficient data for {ticker}: only {len(data)} data points")
        warnings.append(f"Limited data available for {ticker}: only {len(data)} data points. Analysis may be less reliable.")
    
    # Validate required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {', '.join(missing_columns)} in data for {ticker}")
        warnings.append(f"Missing data: {', '.join(missing_columns)}. Analysis may be limited.")
        
        # Create missing columns with default values to allow analysis to proceed
        for col in missing_columns:
            if col == 'Volume':  # Default volume to 0
                data[col] = 0
            elif col in ['Open', 'High', 'Low']:  # Use Close for missing price columns
                if 'Close' in data.columns:
                    data[col] = data['Close']
                else:
                    # If even Close is missing, we can't proceed
                    logger.error(f"Critical data missing for {ticker}: no price data available")
                    errors.append("No price data available for analysis.")
                    return None, warnings, errors, actual_period, actual_interval
    
    return data, warnings, errors, actual_period, actual_interval

def get_ticker_info(ticker_symbol):
    """Get ticker information from Yahoo Finance"""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        if info:
            # Extract useful fields if they exist
            useful_fields = ['shortName', 'longName', 'sector', 'industry', 
                            'exchange', 'currency', 'country', 'market']
            ticker_info = {k: info[k] for k in useful_fields if k in info}
            logger.info(f"Successfully retrieved info for {ticker_symbol}")
            return ticker_info
        return {}
    except Exception as e:
        logger.warning(f"Could not retrieve ticker info: {str(e)}")
        return {}

def find_sector_etf_from_csv(ticker_symbol):
    """
    Search through SPDR sector ETF CSV files to find which sector the ticker belongs to.
    
    Args:
        ticker_symbol (str): The ticker symbol to search for
        
    Returns:
        tuple: (sector_name, etf_symbol) if found, otherwise (None, None)
    """
    logger.info(f"Searching for {ticker_symbol} in SPDR sector ETF CSV files")
    
    # Map of ETF symbols to sector names
    etf_to_sector = {
        'XLF': 'Financials',
        'XLK': 'Information Technology',
        'XLV': 'Health Care',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLB': 'Materials',
        'XLI': 'Industrials',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities',
        'XLC': 'Communication Services'
    }
    
    # List of ETF CSV files to search
    etf_files = [f"{etf}.csv" for etf in etf_to_sector.keys()]
    
    # Normalize the ticker for comparison
    normalized_ticker = ticker_symbol.upper().strip()
    
    # Search through each CSV file
    for etf_file in etf_files:
        try:
            # Check if file exists
            if not os.path.exists(etf_file):
                logger.debug(f"File {etf_file} does not exist, skipping")
                continue
                
            logger.debug(f"Searching in {etf_file}")
            
            # Read the CSV file
            df = pd.read_csv(etf_file, encoding='utf-8')
            
            # Look for the ticker in all columns as the format could vary
            ticker_found = False
            for column in df.columns:
                # Convert to string to handle non-string columns
                if df[column].astype(str).str.contains(normalized_ticker, case=False, regex=False).any():
                    ticker_found = True
                    break
            
            if ticker_found:
                etf_symbol = etf_file.replace('.csv', '')
                sector_name = etf_to_sector.get(etf_symbol)
                logger.info(f"Found {ticker_symbol} in {etf_file}, sector: {sector_name}")
                return sector_name, etf_symbol
                
        except Exception as e:
            logger.warning(f"Error searching in {etf_file}: {str(e)}")
    
    logger.info(f"Ticker {ticker_symbol} not found in any sector ETF CSV file")
    return None, None
