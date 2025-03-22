import pandas as pd
import numpy as np
import datetime
import logging
import traceback
import yfinance as yf

# Import utility modules
from utils.data_utils import (
    normalize_ticker, is_index, get_safe_series, get_ticker_info, 
    download_ticker_data, find_sector_etf_from_csv
)
from utils.indicator_utils import (
    calculate_indicators, identify_support_resistance, 
    identify_weinstein_phase, generate_recommendation, 
    generate_detailed_analysis
)
from utils.chart_utils import create_interactive_chart, create_volume_profile
from utils.backtest_utils import perform_simplified_backtest, create_backtest_charts

logger = logging.getLogger('WeinsteinAnalyzer')

class WeinsteinTickerAnalyzer:
    def __init__(self):
        """Initializes the Weinstein Ticker Analyzer"""
        self.data = None
        self.ticker_symbol = None
        self.period = "1y"
        self.interval = "1wk"
        self.indicators = {}
        self.phase = 0
        self.phase_desc = ""
        self.recommendation = ""
        self.detailed_analysis = ""
        self.last_price = None
        self.market_context = None
        self.sector_data = None
        self.support_resistance_levels = []
        self.errors = []
        self.warnings = []
        self.ticker_info = {}
        self.backtest_results = None
        
    def load_data(self, ticker, period="1y", interval="1wk"):
        """Load data for a specific ticker with enhanced error handling"""
        logger.info(f"Loading data for {ticker} with period={period}, interval={interval}")
        self.ticker_symbol = ticker
        self.period = period
        self.interval = interval
        self.errors = []
        self.warnings = []
        
        # Clear previous data
        self.data = None
        self.phase = 0
        self.phase_desc = ""
        self.recommendation = ""
        self.detailed_analysis = ""
        self.last_price = None
        self.market_context = None
        self.sector_data = None
        self.support_resistance_levels = []
        self.backtest_results = None
        
        try:
            # Download data with error handling 
            self.data, new_warnings, errors, actual_period, actual_interval = download_ticker_data(
                ticker, period, interval
            )
            
            # Update warnings and errors
            self.warnings.extend(new_warnings)
            if errors:
                self.errors.extend(errors)
                return False
                
            # Update instance variables with actual values used
            self.period = actual_period
            self.interval = actual_interval
            
            # Try to get ticker info for better context
            self.ticker_info = get_ticker_info(normalize_ticker(ticker))
            
            # Safe extraction of last price
            try:
                if isinstance(self.data['Close'].iloc[-1], (pd.Series, pd.DataFrame)):
                    self.last_price = float(self.data['Close'].iloc[-1].iloc[0])
                else:
                    self.last_price = float(self.data['Close'].iloc[-1])
            except Exception as e:
                logger.error(f"Error extracting last price: {str(e)}")
                self.warnings.append("Could not determine the last price.")
                self.last_price = None
                
            # Calculate all indicators
            self.data = calculate_indicators(self.data)
            
            # Find support and resistance levels
            self.support_resistance_levels = identify_support_resistance(self.data, self.last_price)
            
            # Load market context - skip for indices to avoid self-comparison
            normalized_ticker = normalize_ticker(ticker)
            if not is_index(normalized_ticker, self.ticker_info):
                self.load_market_context()
            
            # Identify the phase
            self.phase, self.phase_desc = identify_weinstein_phase(self.data)
            
            # Generate recommendation
            self.recommendation = generate_recommendation(
                self.data, self.phase, self.phase_desc, 
                self.market_context, self.sector_data
            )
            
            # Generate detailed analysis text
            self.detailed_analysis = generate_detailed_analysis(
                self.ticker_symbol, self.phase, self.phase_desc, 
                self.recommendation, self.data, self.last_price,
                self.market_context, self.sector_data, 
                self.support_resistance_levels
            )
            
            # Perform backtest automatically if we have enough data
            if len(self.data) >= 30:
                self.backtest_results = perform_simplified_backtest(self.data, self.ticker_symbol)
                if self.backtest_results["success"]:
                    logger.info(f"Backtest completed automatically with {self.backtest_results['total_trades']} trades")
                else:
                    logger.warning(f"Automatic backtest failed: {self.backtest_results.get('error', 'Unknown error')}")
            else:
                self.backtest_results = {
                    "success": False,
                    "error": f"Insufficient data for backtest. Need at least 30 data points, found {len(self.data)}."
                }
                logger.warning(f"Not enough data for automatic backtest: {len(self.data)} data points")
            
            logger.info(f"Successfully loaded and analyzed data for {ticker}")
            return True
                
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            traceback.print_exc()
            self.errors.append(f"Error analyzing {ticker}: {str(e)}")
            return False
    
    def load_market_context(self):
        """Load market context data (S&P 500 index) for relative analysis with enhanced sector detection"""
        try:
            market_data = yf.download(
                "^GSPC",  # S&P 500 index
                period=self.period,
                interval=self.interval,
                progress=False
            )
            
            if len(market_data) > 0:
                # Calculate market indicators
                market_data['MA30'] = market_data['Close'].rolling(window=30).mean()
                market_data['MA30_Slope'] = market_data['MA30'].diff()
                
                # Determine market phase - Use scalar values to avoid Series comparison issues
                current = market_data.iloc[-1]
                
                # Ensure we have scalar values, not Series
                try:
                    current_close = float(current['Close']) if not pd.isna(current['Close']).any() else 0
                    current_ma30 = float(current['MA30']) if not pd.isna(current['MA30']).any() else 0
                    current_ma30_slope = float(current['MA30_Slope']) if not pd.isna(current['MA30_Slope']).any() else 0
                except Exception:
                    # Handle potential Series objects
                    if isinstance(current['Close'], pd.Series):
                        current_close = float(current['Close'].iloc[0]) if not current['Close'].empty and not pd.isna(current['Close'].iloc[0]) else 0
                    else:
                        current_close = float(current['Close']) if not pd.isna(current['Close']) else 0
                        
                    if isinstance(current['MA30'], pd.Series):
                        current_ma30 = float(current['MA30'].iloc[0]) if not current['MA30'].empty and not pd.isna(current['MA30'].iloc[0]) else 0
                    else:
                        current_ma30 = float(current['MA30']) if not pd.isna(current['MA30']) else 0
                        
                    if isinstance(current['MA30_Slope'], pd.Series):
                        current_ma30_slope = float(current['MA30_Slope'].iloc[0]) if not current['MA30_Slope'].empty and not pd.isna(current['MA30_Slope'].iloc[0]) else 0
                    else:
                        current_ma30_slope = float(current['MA30_Slope']) if not pd.isna(current['MA30_Slope']) else 0
                
                price_above_ma = current_close > current_ma30
                ma_slope_positive = current_ma30_slope > 0
                
                if price_above_ma and ma_slope_positive:
                    market_phase = 2  # Uptrend
                elif price_above_ma and not ma_slope_positive:
                    market_phase = 3  # Top formation
                elif not price_above_ma and not ma_slope_positive:
                    market_phase = 4  # Downtrend
                else:
                    market_phase = 1  # Base formation
                
                # Get market performance metrics
                if len(market_data) >= 4:  # At least 4 weeks of data
                    market_1month_perf = (float(market_data['Close'].iloc[-1]) / float(market_data['Close'].iloc[-4]) - 1) * 100
                else:
                    market_1month_perf = 0
                
                # Store market context
                self.market_context = {
                    'phase': market_phase,
                    'last_close': float(market_data['Close'].iloc[-1]),
                    'performance_1month': market_1month_perf
                }
                
                logger.info(f"Market context loaded: Phase {market_phase}")
                
                # Try to load sector data
                self._load_sector_data()
                
            else:
                logger.warning("No market context data available")
                self.market_context = None
        
        except Exception as e:
            logger.warning(f"Error loading market context: {str(e)}")
            self.market_context = None
            
    def _load_sector_data(self):
        """Load sector data for the current ticker"""
        try:
            # STRATEGY 1: First look in CSV files of ETF holdings
            sector, sector_etf = find_sector_etf_from_csv(self.ticker_symbol)
            
            # If not found in CSV files, try alternative methods
            if not sector or not sector_etf:
                logger.info(f"Ticker {self.ticker_symbol} not found in sector ETF CSV files, trying alternative methods")
                
                # Get ticker info with detailed logging
                ticker_obj = yf.Ticker(self.ticker_symbol)
                ticker_info = ticker_obj.info
                
                # STRATEGY 2: Manually map common tickers to sectors
                manual_sector_map = {
                    'AAPL': ('Information Technology', 'XLK'),
                    'MSFT': ('Information Technology', 'XLK'),
                    'AMZN': ('Consumer Discretionary', 'XLY'),
                    'GOOG': ('Communication Services', 'XLC'),
                    'GOOGL': ('Communication Services', 'XLC'),
                    'META': ('Communication Services', 'XLC'),
                    'TSLA': ('Consumer Discretionary', 'XLY'),
                    'JPM': ('Financials', 'XLF'),
                    'V': ('Financials', 'XLF'),
                    'NVDA': ('Information Technology', 'XLK'),
                    'DIS': ('Communication Services', 'XLC')
                    # Add more common tickers as needed
                }
                
                # Check if we have a manual mapping for this ticker
                if self.ticker_symbol.upper() in manual_sector_map:
                    sector, sector_etf = manual_sector_map[self.ticker_symbol.upper()]
                    logger.info(f"Using manual sector mapping for {self.ticker_symbol}: {sector} -> {sector_etf}")
                
                # STRATEGY 3: Try to get sector directly from ticker_info
                elif ticker_info:
                    # Try multiple possible keys where sector info might be stored
                    sector_keys = ['sector', 'sectorDisp', 'sectorChain', 'industryGroup']
                    for key in sector_keys:
                        if key in ticker_info and ticker_info[key]:
                            sector = ticker_info[key]
                            logger.info(f"Found sector '{sector}' using key '{key}'")
                            break
                    
                    # If no sector found, try industry as fallback
                    if not sector:
                        industry_keys = ['industry', 'industryDisp', 'industryKey']
                        for key in industry_keys:
                            if key in ticker_info and ticker_info[key]:
                                industry = ticker_info[key]
                                logger.info(f"No sector found, using industry: {industry}")
                                
                                # Map industry to sector (simplified mapping)
                                industry_to_sector = {
                                    'software': 'Information Technology',
                                    'hardware': 'Information Technology',
                                    'semiconductor': 'Information Technology',
                                    'bank': 'Financials',
                                    'insurance': 'Financials',
                                    'pharma': 'Health Care',
                                    'biotech': 'Health Care',
                                    'medical': 'Health Care',
                                    'retail': 'Consumer Discretionary',
                                    'auto': 'Consumer Discretionary',
                                    'aerospace': 'Industrials',
                                    'defense': 'Industrials',
                                    'telecom': 'Communication Services',
                                    'media': 'Communication Services',
                                    'food': 'Consumer Staples',
                                    'beverage': 'Consumer Staples',
                                    'oil': 'Energy',
                                    'gas': 'Energy',
                                    'chemical': 'Materials',
                                    'mining': 'Materials',
                                    'real estate': 'Real Estate',
                                    'reit': 'Real Estate',
                                    'utility': 'Utilities',
                                    'electric': 'Utilities'
                                }
                                
                                # Try to match industry to a sector
                                industry_lower = industry.lower()
                                for ind_key, sec_value in industry_to_sector.items():
                                    if ind_key in industry_lower:
                                        sector = sec_value
                                        logger.info(f"Mapped industry '{industry}' to sector '{sector}'")
                                        break
                                
                                if sector:
                                    break
            
            # Sector to ETF mapping
            sector_etfs = {
                # Standard GICS sectors
                'Information Technology': 'XLK',
                'Financials': 'XLF',
                'Health Care': 'XLV',
                'Consumer Discretionary': 'XLY',
                'Industrials': 'XLI',
                'Communication Services': 'XLC',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU',
                
                # Alternative/legacy sector names from Yahoo Finance
                'Technology': 'XLK',
                'Financial Services': 'XLF',
                'Healthcare': 'XLV',
                'Consumer Cyclical': 'XLY',
                'Communication': 'XLC',
                'Consumer Defensive': 'XLP',
                'Basic Materials': 'XLB',
                'Financial': 'XLF'
            }
            
            # If we have a sector but no ETF yet, look it up in our mapping
            if sector and not sector_etf:
                if sector in sector_etfs:
                    sector_etf = sector_etfs[sector]
                    logger.info(f"Mapped sector '{sector}' to ETF '{sector_etf}'")
                else:
                    # Try case-insensitive matching
                    sector_lower = sector.lower()
                    for sec_key, etf in sector_etfs.items():
                        if sec_key.lower() == sector_lower:
                            sector_etf = etf
                            logger.info(f"Case-insensitive match: '{sector}' to '{sec_key}' (ETF: {etf})")
                            sector = sec_key  # Use the correctly cased sector name
                            break
            
            # Only proceed with sector analysis if we have both sector and ETF
            if sector and sector_etf:
                logger.info(f"Downloading data for sector ETF: {sector_etf}")
                
                # Download sector ETF data
                sector_data = yf.download(
                    sector_etf,
                    period=self.period,
                    interval=self.interval,
                    progress=False
                )
                
                if len(sector_data) > 0:
                    # Calculate sector indicators
                    sector_data['MA30'] = sector_data['Close'].rolling(window=30).mean()
                    sector_data['MA30_Slope'] = sector_data['MA30'].diff()
                    
                    # Determine sector phase
                    current_sector = sector_data.iloc[-1]
                    
                    # Safe extraction of scalar values
                    try:
                        sector_close = float(current_sector['Close']) if not pd.isna(current_sector['Close']).any() else 0
                        sector_ma30 = float(current_sector['MA30']) if not pd.isna(current_sector['MA30']).any() else 0
                        sector_ma30_slope = float(current_sector['MA30_Slope']) if not pd.isna(current_sector['MA30_Slope']).any() else 0
                    except Exception as e:
                        logger.warning(f"Error extracting sector values: {str(e)}")
                        # Fallback method
                        if isinstance(current_sector['Close'], pd.Series):
                            sector_close = float(current_sector['Close'].iloc[0]) if not current_sector['Close'].empty else 0
                        else:
                            sector_close = float(current_sector['Close']) if not pd.isna(current_sector['Close']) else 0
                            
                        if isinstance(current_sector['MA30'], pd.Series):
                            sector_ma30 = float(current_sector['MA30'].iloc[0]) if not current_sector['MA30'].empty else 0
                        else:
                            sector_ma30 = float(current_sector['MA30']) if not pd.isna(current_sector['MA30']) else 0
                            
                        if isinstance(current_sector['MA30_Slope'], pd.Series):
                            sector_ma30_slope = float(current_sector['MA30_Slope'].iloc[0]) if not current_sector['MA30_Slope'].empty and not pd.isna(current_sector['MA30_Slope'].iloc[0]) else 0
                        else:
                            sector_ma30_slope = float(current_sector['MA30_Slope']) if not pd.isna(current_sector['MA30_Slope']) else 0
                    
                    sector_price_above_ma = sector_close > sector_ma30
                    sector_ma_slope_positive = sector_ma30_slope > 0
                    
                    if sector_price_above_ma and sector_ma_slope_positive:
                        sector_phase = 2  # Uptrend
                    elif sector_price_above_ma and not sector_ma_slope_positive:
                        sector_phase = 3  # Top formation
                    elif not sector_price_above_ma and not sector_ma_slope_positive:
                        sector_phase = 4  # Downtrend
                    else:
                        sector_phase = 1  # Base formation
                    
                    # Get sector performance metrics
                    if len(sector_data) >= 4:
                        sector_1month_perf = (float(sector_data['Close'].iloc[-1]) / float(sector_data['Close'].iloc[-4]) - 1) * 100
                    else:
                        sector_1month_perf = 0
                    
                    # Calculate relative strength vs market
                    market_data = yf.download("^GSPC", period=self.period, interval=self.interval, progress=False)
                    
                    if len(market_data) == len(sector_data):
                        relative_strength = (float(sector_data['Close'].iloc[-1]) / float(sector_data['Close'].iloc[0])) / \
                                           (float(market_data['Close'].iloc[-1]) / float(market_data['Close'].iloc[0])) * 100 - 100
                    else:
                        # Handle mismatched lengths by using common date range
                        logger.warning("Market and sector data lengths don't match. Using common date range for RS calculation.")
                        common_start = max(market_data.index[0], sector_data.index[0])
                        common_end = min(market_data.index[-1], sector_data.index[-1])
                        
                        market_start = float(market_data.loc[market_data.index >= common_start, 'Close'].iloc[0])
                        market_end = float(market_data.loc[market_data.index <= common_end, 'Close'].iloc[-1])
                        sector_start = float(sector_data.loc[sector_data.index >= common_start, 'Close'].iloc[0])
                        sector_end = float(sector_data.loc[sector_data.index <= common_end, 'Close'].iloc[-1])
                        
                        market_perf = market_end / market_start
                        sector_perf = sector_end / sector_start
                        
                        if market_perf > 0:
                            relative_strength = (sector_perf / market_perf) * 100 - 100
                        else:
                            relative_strength = 0
                    
                    # Store sector context
                    self.sector_data = {
                        'name': sector,
                        'etf': sector_etf,
                        'phase': sector_phase,
                        'last_close': float(sector_data['Close'].iloc[-1]),
                        'performance_1month': sector_1month_perf,
                        'relative_strength': relative_strength
                    }
                    
                    logger.info(f"Sector data loaded: {sector} (Phase {sector_phase})")
                else:
                    logger.warning(f"Downloaded empty dataset for sector ETF {sector_etf}")
                    self.sector_data = None
            else:
                logger.warning(f"Skipping sector analysis for {self.ticker_symbol}: no valid sector or ETF determined")
                self.sector_data = None
                    
        except Exception as e:
            logger.error(f"Error in sector analysis: {str(e)}")
            logger.error(traceback.format_exc())
            self.sector_data = None
    
    def create_interactive_chart(self):
        """Create an interactive chart with Plotly and enhanced visualization"""
        return create_interactive_chart(
            self.data, 
            self.ticker_symbol, 
            self.phase, 
            self.phase_desc, 
            self.recommendation, 
            self.support_resistance_levels, 
            self.last_price
        )
    
    def create_volume_profile(self, lookback_period=None):
        """Create a volume profile for the specified timeframe"""
        return create_volume_profile(
            self.data, 
            self.ticker_symbol, 
            self.last_price, 
            lookback_period
        )
    
    def create_simplified_backtest_charts(self, backtest_results=None):
        """Create simplified charts for backtesting results"""
        # Use passed backtest_results or self.backtest_results
        if backtest_results is None:
            backtest_results = self.backtest_results
            
        if backtest_results is None:
            # If still None, run the backtest first
            self.backtest_results = perform_simplified_backtest(self.data, self.ticker_symbol)
            backtest_results = self.backtest_results
        
        # Get price data for Buy & Hold comparison
        price_data = None
        if self.data is not None and 'Close' in self.data.columns:
            price_data = get_safe_series(self.data, 'Close')
            
        # Create charts
        from utils.backtest_utils import create_backtest_charts
        return create_backtest_charts(backtest_results, price_data)
