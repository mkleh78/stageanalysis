import pandas as pd
import numpy as np
import datetime
import logging
import traceback
import yfinance as yf
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field
from functools import lru_cache
from contextlib import contextmanager

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


@dataclass
class AnalysisResult:
    """Strukturiertes Ergebnis der Weinstein-Analyse"""
    success: bool
    ticker: str
    phase: int
    phase_description: str
    recommendation: str
    detailed_analysis: str
    last_price: Optional[float]
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    support_resistance_data: List[Dict[str, float]] = field(default_factory=list)
    market_context: Optional[Dict[str, Any]] = None
    sector_data: Optional[Dict[str, Any]] = None
    backtest_results: Optional[Dict[str, Any]] = None
    ticker_info: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class WeinsteinTickerAnalyzer:
    """
    Analyzes stocks using the Weinstein method with enhanced error handling,
    caching, and structured output.
    """
    
    # Konstanten
    MIN_BACKTEST_DATAPOINTS = 30
    DEFAULT_MA_PERIOD = 30
    MONTH_IN_WEEKS = 4
    
    # Valid input values
    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    
    # Sector ETF mapping
    SECTOR_ETF_MAP = {
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
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Cyclical': 'XLY',
        'Communication': 'XLC',
        'Consumer Defensive': 'XLP',
        'Basic Materials': 'XLB',
        'Financial': 'XLF'
    }
    
    # Manual ticker to sector mapping for common stocks
    MANUAL_TICKER_SECTOR_MAP = {
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
    }
    
    # Industry to sector mapping
    INDUSTRY_TO_SECTOR_MAP = {
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
    
    def __init__(self, data_provider=None, logger_instance=None):
        """
        Initializes the Weinstein Ticker Analyzer
        
        Args:
            data_provider: Data provider (default: yfinance)
            logger_instance: Logger instance (default: module logger)
        """
        self.data_provider = data_provider or yf
        self.logger = logger_instance or logger
        
        # Data attributes
        self.data: Optional[pd.DataFrame] = None
        self.ticker_symbol: Optional[str] = None
        self.period: str = "1y"
        self.interval: str = "1wk"
        
        # Analysis results
        self.indicators: Dict[str, pd.Series] = {}
        self.phase: int = 0
        self.phase_desc: str = ""
        self.recommendation: str = ""
        self.detailed_analysis: str = ""
        self.last_price: Optional[float] = None
        self.market_context: Optional[Dict[str, Any]] = None
        self.sector_data: Optional[Dict[str, Any]] = None
        self.support_resistance_levels: List[Dict[str, float]] = []
        self.ticker_info: Dict[str, Any] = {}
        self.backtest_results: Optional[Dict[str, Any]] = None
        
        # Status tracking
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Caching
        self._market_data_cache: Dict[str, pd.DataFrame] = {}
        self._sector_etf_cache: Dict[str, pd.DataFrame] = {}
    
    def _extract_scalar_value(self, value: Union[pd.Series, float, Any], default: float = 0) -> float:
        """
        Extrahiert einen Skalarwert aus Series oder einzelnem Wert
        
        Args:
            value: Der zu extrahierende Wert
            default: Standardwert bei Fehler
            
        Returns:
            float: Extrahierter Skalarwert
        """
        try:
            if isinstance(value, pd.Series):
                return float(value.iloc[0]) if not value.empty and not pd.isna(value.iloc[0]) else default
            else:
                return float(value) if not pd.isna(value) else default
        except Exception as e:
            self.logger.debug(f"Error extracting scalar value: {str(e)}")
            return default
    
    def _validate_inputs(self, ticker: str, period: str, interval: str) -> Tuple[str, str, str]:
        """
        Validiert die Eingabeparameter
        
        Args:
            ticker: Ticker Symbol
            period: Zeitperiode
            interval: Datenintervall
            
        Returns:
            Tuple[str, str, str]: Validierte Parameter
        """
        # Ticker validation
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker symbol provided")
        
        ticker = ticker.strip().upper()
        
        # Period validation
        if period not in self.VALID_PERIODS:
            self.warnings.append(f"Invalid period '{period}', using default '1y'")
            period = "1y"
        
        # Interval validation
        if interval not in self.VALID_INTERVALS:
            self.warnings.append(f"Invalid interval '{interval}', using default '1wk'")
            interval = "1wk"
        
        return ticker, period, interval
    
    @contextmanager
    def analysis_context(self, ticker: str):
        """
        Context manager für Analyse-Sessions
        
        Args:
            ticker: Ticker Symbol
        """
        try:
            self.logger.info(f"Starting analysis for {ticker}")
            yield self
        except Exception as e:
            self.logger.error(f"Analysis failed for {ticker}: {str(e)}")
            self.errors.append(f"Analysis failed: {str(e)}")
            raise
        finally:
            self.logger.info(f"Completed analysis for {ticker}")
    
    def load_data(self, ticker: str, period: str = "1y", interval: str = "1wk") -> bool:
        """
        Hauptmethode zum Laden und Analysieren der Daten
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for data (default: "1y")
            interval: Data interval (default: "1wk")
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self.analysis_context(ticker):
            # Initialisierung
            self._initialize_analysis(ticker, period, interval)
            
            # Daten laden
            if not self._fetch_ticker_data():
                return False
            
            # Analyse durchführen
            self._calculate_indicators()
            self._load_market_and_sector_context()
            self._analyze_phase()
            self._generate_recommendations()
            self._run_backtest()
            
            return True
    
    def _initialize_analysis(self, ticker: str, period: str, interval: str) -> None:
        """
        Initialisiert die Analyse-Parameter und setzt den Zustand zurück
        
        Args:
            ticker: Ticker Symbol
            period: Zeitperiode
            interval: Datenintervall
        """
        # Validierung
        ticker, period, interval = self._validate_inputs(ticker, period, interval)
        
        # Parameter setzen
        self.ticker_symbol = ticker
        self.period = period
        self.interval = interval
        
        # Reset state
        self.errors = []
        self.warnings = []
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
        self.indicators = {}
    
    def _fetch_ticker_data(self) -> bool:
        """
        Lädt die Ticker-Daten
        
        Returns:
            bool: True wenn erfolgreich, False sonst
        """
        try:
            # Download data with error handling
            self.data, new_warnings, errors, actual_period, actual_interval = download_ticker_data(
                self.ticker_symbol, self.period, self.interval
            )
            
            # Update warnings and errors
            self.warnings.extend(new_warnings)
            if errors:
                self.errors.extend(errors)
                return False
            
            # Update instance variables with actual values used
            self.period = actual_period
            self.interval = actual_interval
            
            # Get ticker info
            self.ticker_info = get_ticker_info(normalize_ticker(self.ticker_symbol))
            
            # Extract last price
            self._extract_last_price()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {self.ticker_symbol}: {str(e)}")
            self.errors.append(f"Error fetching data: {str(e)}")
            return False
    
    def _extract_last_price(self) -> None:
        """Extrahiert den letzten Preis aus den Daten"""
        try:
            if self.data is not None and 'Close' in self.data.columns:
                close_series = self.data['Close']
                if isinstance(close_series.iloc[-1], (pd.Series, pd.DataFrame)):
                    self.last_price = float(close_series.iloc[-1].iloc[0])
                else:
                    self.last_price = float(close_series.iloc[-1])
            else:
                self.last_price = None
                self.warnings.append("Could not determine the last price.")
        except Exception as e:
            self.logger.error(f"Error extracting last price: {str(e)}")
            self.warnings.append("Could not determine the last price.")
            self.last_price = None
    
    def _calculate_indicators(self) -> None:
        """Berechnet technische Indikatoren"""
        if self.data is not None:
            self.data = calculate_indicators(self.data)
            self.support_resistance_levels = identify_support_resistance(self.data, self.last_price)
    
    def _load_market_and_sector_context(self) -> None:
        """Lädt Markt- und Sektor-Kontext"""
        # Skip for indices to avoid self-comparison
        normalized_ticker = normalize_ticker(self.ticker_symbol)
        if not is_index(normalized_ticker, self.ticker_info):
            self._load_market_context()
            self._load_sector_data()
    
    @lru_cache(maxsize=32)
    def _get_cached_market_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Cached market data retrieval
        
        Args:
            symbol: Market symbol
            period: Time period
            interval: Data interval
            
        Returns:
            Optional[pd.DataFrame]: Market data or None
        """
        try:
            return self.data_provider.download(
                symbol,
                period=period,
                interval=interval,
                progress=False
            )
        except Exception as e:
            self.logger.error(f"Error downloading {symbol}: {str(e)}")
            return None
    
    def _load_market_context(self) -> None:
        """Lädt Marktkontext-Daten (S&P 500 Index) für relative Analyse"""
        try:
            market_data = self._get_cached_market_data("^GSPC", self.period, self.interval)
            
            if market_data is not None and len(market_data) > 0:
                # Calculate market indicators
                market_data['MA30'] = market_data['Close'].rolling(window=self.DEFAULT_MA_PERIOD).mean()
                market_data['MA30_Slope'] = market_data['MA30'].diff()
                
                # Determine market phase
                current = market_data.iloc[-1]
                
                current_close = self._extract_scalar_value(current['Close'])
                current_ma30 = self._extract_scalar_value(current['MA30'])
                current_ma30_slope = self._extract_scalar_value(current['MA30_Slope'])
                
                price_above_ma = current_close > current_ma30
                ma_slope_positive = current_ma30_slope > 0
                
                # Determine phase
                if price_above_ma and ma_slope_positive:
                    market_phase = 2  # Uptrend
                elif price_above_ma and not ma_slope_positive:
                    market_phase = 3  # Top formation
                elif not price_above_ma and not ma_slope_positive:
                    market_phase = 4  # Downtrend
                else:
                    market_phase = 1  # Base formation
                
                # Get market performance metrics
                market_1month_perf = 0
                if len(market_data) >= self.MONTH_IN_WEEKS:
                    market_1month_perf = (
                        float(market_data['Close'].iloc[-1]) / 
                        float(market_data['Close'].iloc[-self.MONTH_IN_WEEKS]) - 1
                    ) * 100
                
                # Store market context
                self.market_context = {
                    'phase': market_phase,
                    'last_close': float(market_data['Close'].iloc[-1]),
                    'performance_1month': market_1month_perf
                }
                
                self.logger.info(f"Market context loaded: Phase {market_phase}")
            else:
                self.logger.warning("No market context data available")
                self.market_context = None
                
        except Exception as e:
            self.logger.warning(f"Error loading market context: {str(e)}")
            self.market_context = None
    
    def _load_sector_data(self) -> None:
        """Lädt Sektordaten für den aktuellen Ticker"""
        try:
            sector, sector_etf = self._identify_sector_and_etf()
            
            if sector and sector_etf:
                self._analyze_sector_performance(sector, sector_etf)
            else:
                self.logger.warning(
                    f"Skipping sector analysis for {self.ticker_symbol}: "
                    f"no valid sector or ETF determined"
                )
                self.sector_data = None
                
        except Exception as e:
            self.logger.error(f"Error in sector analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.sector_data = None
    
    def _identify_sector_and_etf(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Identifiziert Sektor und zugehörigen ETF für den Ticker
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (Sektor, ETF Symbol)
        """
        # Strategy 1: Check CSV files
        sector, sector_etf = find_sector_etf_from_csv(self.ticker_symbol)
        
        if sector and sector_etf:
            return sector, sector_etf
        
        # Strategy 2: Manual mapping
        ticker_upper = self.ticker_symbol.upper()
        if ticker_upper in self.MANUAL_TICKER_SECTOR_MAP:
            return self.MANUAL_TICKER_SECTOR_MAP[ticker_upper]
        
        # Strategy 3: Get from ticker info
        sector = self._extract_sector_from_ticker_info()
        
        if sector:
            # Map sector to ETF
            sector_etf = self._map_sector_to_etf(sector)
            if sector_etf:
                return sector, sector_etf
        
        return None, None
    
    def _extract_sector_from_ticker_info(self) -> Optional[str]:
        """
        Extrahiert Sektor-Information aus Ticker-Info
        
        Returns:
            Optional[str]: Sektor oder None
        """
        try:
            ticker_obj = self.data_provider.Ticker(self.ticker_symbol)
            ticker_info = ticker_obj.info
            
            # Try sector keys
            sector_keys = ['sector', 'sectorDisp', 'sectorChain', 'industryGroup']
            for key in sector_keys:
                if key in ticker_info and ticker_info[key]:
                    return ticker_info[key]
            
            # Try industry as fallback
            industry_keys = ['industry', 'industryDisp', 'industryKey']
            for key in industry_keys:
                if key in ticker_info and ticker_info[key]:
                    industry = ticker_info[key]
                    # Map industry to sector
                    return self._map_industry_to_sector(industry)
                    
        except Exception as e:
            self.logger.debug(f"Error extracting sector from ticker info: {str(e)}")
        
        return None
    
    def _map_industry_to_sector(self, industry: str) -> Optional[str]:
        """
        Mappt Industry zu Sektor
        
        Args:
            industry: Industry Name
            
        Returns:
            Optional[str]: Sektor oder None
        """
        industry_lower = industry.lower()
        for ind_key, sec_value in self.INDUSTRY_TO_SECTOR_MAP.items():
            if ind_key in industry_lower:
                return sec_value
        return None
    
    def _map_sector_to_etf(self, sector: str) -> Optional[str]:
        """
        Mappt Sektor zu ETF Symbol
        
        Args:
            sector: Sektor Name
            
        Returns:
            Optional[str]: ETF Symbol oder None
        """
        # Direct mapping
        if sector in self.SECTOR_ETF_MAP:
            return self.SECTOR_ETF_MAP[sector]
        
        # Case-insensitive matching
        sector_lower = sector.lower()
        for sec_key, etf in self.SECTOR_ETF_MAP.items():
            if sec_key.lower() == sector_lower:
                return etf
        
        return None
    
    def _analyze_sector_performance(self, sector: str, sector_etf: str) -> None:
        """
        Analysiert die Performance des Sektors
        
        Args:
            sector: Sektor Name
            sector_etf: ETF Symbol für den Sektor
        """
        self.logger.info(f"Downloading data for sector ETF: {sector_etf}")
        
        sector_data = self._get_cached_market_data(sector_etf, self.period, self.interval)
        
        if sector_data is None or len(sector_data) == 0:
            self.logger.warning(f"No data available for sector ETF {sector_etf}")
            return
        
        # Calculate sector indicators
        sector_data['MA30'] = sector_data['Close'].rolling(window=self.DEFAULT_MA_PERIOD).mean()
        sector_data['MA30_Slope'] = sector_data['MA30'].diff()
        
        # Determine sector phase
        current_sector = sector_data.iloc[-1]
        
        sector_close = self._extract_scalar_value(current_sector['Close'])
        sector_ma30 = self._extract_scalar_value(current_sector['MA30'])
        sector_ma30_slope = self._extract_scalar_value(current_sector['MA30_Slope'])
        
        sector_price_above_ma = sector_close > sector_ma30
        sector_ma_slope_positive = sector_ma30_slope > 0
        
        # Determine phase
        if sector_price_above_ma and sector_ma_slope_positive:
            sector_phase = 2  # Uptrend
        elif sector_price_above_ma and not sector_ma_slope_positive:
            sector_phase = 3  # Top formation
        elif not sector_price_above_ma and not sector_ma_slope_positive:
            sector_phase = 4  # Downtrend
        else:
            sector_phase = 1  # Base formation
        
        # Get sector performance metrics
        sector_1month_perf = 0
        if len(sector_data) >= self.MONTH_IN_WEEKS:
            sector_1month_perf = (
                float(sector_data['Close'].iloc[-1]) / 
                float(sector_data['Close'].iloc[-self.MONTH_IN_WEEKS]) - 1
            ) * 100
        
        # Calculate relative strength vs market
        relative_strength = self._calculate_relative_strength(sector_data)
        
        # Store sector context
        self.sector_data = {
            'name': sector,
            'etf': sector_etf,
            'phase': sector_phase,
            'last_close': float(sector_data['Close'].iloc[-1]),
            'performance_1month': sector_1month_perf,
            'relative_strength': relative_strength
        }
        
        self.logger.info(f"Sector data loaded: {sector} (Phase {sector_phase})")
    
    def _calculate_relative_strength(self, sector_data: pd.DataFrame) -> float:
        """
        Berechnet die relative Stärke des Sektors zum Markt
        
        Args:
            sector_data: Sektor-Daten
            
        Returns:
            float: Relative Stärke in Prozent
        """
        try:
            market_data = self._get_cached_market_data("^GSPC", self.period, self.interval)
            
            if market_data is None or len(market_data) == 0:
                return 0
            
            if len(market_data) == len(sector_data):
                relative_strength = (
                    (float(sector_data['Close'].iloc[-1]) / float(sector_data['Close'].iloc[0])) /
                    (float(market_data['Close'].iloc[-1]) / float(market_data['Close'].iloc[0]))
                ) * 100 - 100
            else:
                # Handle mismatched lengths
                common_start = max(market_data.index[0], sector_data.index[0])
                common_end = min(market_data.index[-1], sector_data.index[-1])
                
                market_start = float(market_data.loc[market_data.index >= common_start, 'Close'].iloc[0])
                market_end = float(market_data.loc[market_data.index <= common_end, 'Close'].iloc[-1])
                sector_start = float(sector_data.loc[sector_data.index >= common_start, 'Close'].iloc[0])
                sector_end = float(sector_data.loc[sector_data.index <= common_end, 'Close'].iloc[-1])
                
                market_perf = market_end / market_start
                sector_perf = sector_end / sector_start
                
                relative_strength = (sector_perf / market_perf) * 100 - 100 if market_perf > 0 else 0
            
            return relative_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating relative strength: {str(e)}")
            return 0
    
    def _analyze_phase(self) -> None:
        """Analysiert die Weinstein-Phase"""
        if self.data is not None:
            self.phase, self.phase_desc = identify_weinstein_phase(self.data)
    
    def _generate_recommendations(self) -> None:
        """Generiert Handelsempfehlungen und detaillierte Analyse"""
        if self.data is not None:
            # Generate recommendation
            self.recommendation = generate_recommendation(
                self.data, self.phase, self.phase_desc,
                self.market_context, self.sector_data
            )
            
            # Generate detailed analysis
            self.detailed_analysis = generate_detailed_analysis(
                self.ticker_symbol, self.phase, self.phase_desc,
                self.recommendation, self.data, self.last_price,
                self.market_context, self.sector_data,
                self.support_resistance_levels
            )
    
    def _run_backtest(self) -> None:
        """Führt automatisch einen Backtest durch wenn genügend Daten vorhanden sind"""
        if self.data is not None and len(self.data) >= self.MIN_BACKTEST_DATAPOINTS:
            self.backtest_results = perform_simplified_backtest(self.data, self.ticker_symbol)
            if self.backtest_results["success"]:
                self.logger.info(
                    f"Backtest completed automatically with "
                    f"{self.backtest_results['total_trades']} trades"
                )
            else:
                self.logger.warning(
                    f"Automatic backtest failed: "
                    f"{self.backtest_results.get('error', 'Unknown error')}"
                )
        else:
            data_points = len(self.data) if self.data is not None else 0
            self.backtest_results = {
                "success": False,
                "error": (
                    f"Insufficient data for backtest. Need at least "
                    f"{self.MIN_BACKTEST_DATAPOINTS} data points, found {data_points}."
                )
            }
            self.logger.warning(f"Not enough data for automatic backtest: {data_points} data points")
    
    def get_analysis_result(self) -> AnalysisResult:
        """
        Gibt das strukturierte Analyse-Ergebnis zurück
        
        Returns:
            AnalysisResult: Strukturiertes Ergebnis-Objekt
        """
        # Extract support and resistance levels
        support_levels = []
        resistance_levels = []
        
        for level in self.support_resistance_levels:
            if level['type'] == 'support':
                support_levels.append(level['price'])
            else:
                resistance_levels.append(level['price'])
        
        return AnalysisResult(
            success=len(self.errors) == 0,
            ticker=self.ticker_symbol or "",
            phase=self.phase,
            phase_description=self.phase_desc,
            recommendation=self.recommendation,
            detailed_analysis=self.detailed_analysis,
            last_price=self.last_price,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            support_resistance_data=self.support_resistance_levels,
            market_context=self.market_context,
            sector_data=self.sector_data,
            backtest_results=self.backtest_results,
            ticker_info=self.ticker_info,
            errors=self.errors.copy(),
            warnings=self.warnings.copy()
        )
    
    def create_interactive_chart(self) -> Any:
        """
        Erstellt ein interaktives Chart mit Plotly
        
        Returns:
            Plotly Figure Object
        """
        if self.data is None:
            raise ValueError("No data available for charting")
            
        return create_interactive_chart(
            self.data,
            self.ticker_symbol,
            self.phase,
            self.phase_desc,
            self.recommendation,
            self.support_resistance_levels,
            self.last_price
        )
    
    def create_volume_profile(self, lookback_period: Optional[int] = None) -> Any:
        """
        Erstellt ein Volumen-Profil für den angegebenen Zeitraum
        
        Args:
            lookback_period: Anzahl der Perioden für das Profil
            
        Returns:
            Plotly Figure Object
        """
        if self.data is None:
            raise ValueError("No data available for volume profile")
            
        return create_volume_profile(
            self.data,
            self.ticker_symbol,
            self.last_price,
            lookback_period
        )
    
    def create_simplified_backtest_charts(self, 
                                        backtest_results: Optional[Dict[str, Any]] = None) -> Any:
        """
        Erstellt vereinfachte Charts für Backtesting-Ergebnisse
        
        Args:
            backtest_results: Optionale Backtest-Ergebnisse
            
        Returns:
            Plotly Figure Objects
        """
        # Use passed backtest_results or self.backtest_results
        if backtest_results is None:
            backtest_results = self.backtest_results
        
        if backtest_results is None:
            # If still None, run the backtest first
            self._run_backtest()
            backtest_results = self.backtest_results
        
        # Get price data for Buy & Hold comparison
        price_data = None
        if self.data is not None and 'Close' in self.data.columns:
            price_data = get_safe_series(self.data, 'Close')
        
        # Create charts
        return create_backtest_charts(backtest_results, price_data)


# Convenience function für einfache Nutzung
def analyze_ticker(ticker: str, period: str = "1y", 
                  interval: str = "1wk") -> AnalysisResult:
    """
    Convenience function für schnelle Ticker-Analyse
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for data
        interval: Data interval
        
    Returns:
        AnalysisResult: Strukturiertes Analyse-Ergebnis
    """
    analyzer = WeinsteinTickerAnalyzer()
    success = analyzer.load_data(ticker, period, interval)
    
    if not success:
        # Return error result
        return AnalysisResult(
            success=False,
            ticker=ticker,
            phase=0,
            phase_description="",
            recommendation="",
            detailed_analysis="",
            last_price=None,
            errors=analyzer.errors,
            warnings=analyzer.warnings
        )
    
    return analyzer.get_analysis_result()
