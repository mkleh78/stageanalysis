import pandas as pd
import numpy as np
import logging
import traceback
from utils.data_utils import get_safe_series

logger = logging.getLogger('WeinsteinAnalyzer')

def calculate_indicators(df, data_length=None):
    """Calculate technical indicators with enhanced methods and adaptability for limited data"""
    if df is None or len(df) == 0:
        return df
        
    try:
        # If data_length not provided, use actual dataframe length
        if data_length is None:
            data_length = len(df)
        
        # Ensure we're working with Series and not DataFrames
        close_series = get_safe_series(df, 'Close')
        open_series = get_safe_series(df, 'Open')
        high_series = get_safe_series(df, 'High')
        low_series = get_safe_series(df, 'Low')
        volume_series = get_safe_series(df, 'Volume')
        
        # Moving Averages with adaptive windows based on available data
        ma_windows = {
            'MA10': min(10, max(3, data_length // 5)),  # At least 3 periods
            'MA30': min(30, max(5, data_length // 3)),  # At least 5 periods
            'MA50': min(50, max(10, data_length // 2)), # At least 10 periods
            'MA200': min(200, max(20, data_length))     # At least 20 periods
        }
        
        # Calculate MAs with adaptive windows
        df['MA10'] = close_series.rolling(window=ma_windows['MA10']).mean()
        df['MA30'] = close_series.rolling(window=ma_windows['MA30']).mean()
        
        # Only calculate longer MAs if enough data is available
        if data_length >= ma_windows['MA50'] * 1.2:  # Need 20% more data than window size
            df['MA50'] = close_series.rolling(window=ma_windows['MA50']).mean()
        
        if data_length >= ma_windows['MA200'] * 1.1:  # Need 10% more data than window size
            df['MA200'] = close_series.rolling(window=ma_windows['MA200']).mean()
        
        # MA Slopes for trend direction and strength
        df['MA10_Slope'] = df['MA10'].diff()
        df['MA30_Slope'] = df['MA30'].diff()
        
        # Adaptive slope calculation based on available data
        slope_periods = min(4, max(1, data_length // 10))
        df['MA30_Slope_4Wk'] = df['MA30'].diff(slope_periods)
        
        # Distance from MAs as percentage (only calculate if MA exists)
        df['Pct_From_MA30'] = (close_series / df['MA30'] - 1) * 100 if 'MA30' in df else pd.Series(index=df.index)
        
        if 'MA200' in df:
            df['Pct_From_MA200'] = (close_series / df['MA200'] - 1) * 100
        
        # RSI (Relative Strength Index) with adaptive window
        rsi_window = min(14, max(5, data_length // 4))
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_window).mean()
        
        # Prevent division by zero
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rs = rs.fillna(0)
        
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range) for stop loss calculation
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift())
        tr3 = abs(low_series - close_series.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=min(14, data_length // 3)).mean()
        
        # Bollinger Bands (2 standard deviations) with adaptive window
        bb_window = min(30, max(5, data_length // 3))
        df['BBand_Mid'] = df['MA30'] if 'MA30' in df else close_series.rolling(window=bb_window).mean()
        df['BBand_Std'] = close_series.rolling(window=bb_window).std()
        df['BBand_Upper'] = df['BBand_Mid'] + (df['BBand_Std'] * 2)
        df['BBand_Lower'] = df['BBand_Mid'] - (df['BBand_Std'] * 2)
        
        # Bollinger Band Width (volatility indicator)
        df['BB_Width'] = (df['BBand_Upper'] - df['BBand_Lower']) / df['BBand_Mid'] * 100
        
        # Volume analysis - only if volume data is available and not all zeros
        if not volume_series.isna().all() and (volume_series > 0).any():
            vol_window = min(30, max(5, data_length // 3))
            df['VolMA30'] = volume_series.rolling(window=vol_window).mean()
            
            # Handle zero values in volume MA
            vol_ma_mean = df['VolMA30'].mean()
            df['VolMA30'] = df['VolMA30'].replace(0, vol_ma_mean if vol_ma_mean > 0 else 1)
            
            df['Vol_Ratio'] = volume_series / df['VolMA30']
            df['Vol_Ratio'] = df['Vol_Ratio'].fillna(0)
            
            # On-Balance Volume (OBV) - only if we have enough data
            if data_length >= 5:
                obv = pd.Series(0, index=df.index)
                for i in range(1, len(df)):
                    if close_series.iloc[i] > close_series.iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + volume_series.iloc[i]
                    elif close_series.iloc[i] < close_series.iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - volume_series.iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                
                df['OBV'] = obv
                
                if data_length >= 20:
                    df['OBV_MA20'] = df['OBV'].rolling(window=min(20, data_length // 2)).mean()
        else:
            # Create placeholder volume indicators with zeros if volume data not available
            df['VolMA30'] = 0
            df['Vol_Ratio'] = 0
            df['OBV'] = 0
            
            logger.warning("No volume data available. Volume-based indicators will be limited.")
        
        # Price percent from high/low with adaptive window
        if data_length >= 10:
            lookback = min(52, max(data_length // 2, 5))  # At least 5 periods, up to 52
            rolling_high = close_series.rolling(window=lookback).max()
            rolling_low = close_series.rolling(window=lookback).min()
            df['Pct_From_52wk_High'] = (close_series / rolling_high - 1) * 100
            df['Pct_From_52wk_Low'] = (close_series / rolling_low - 1) * 100
        else:
            # For very limited data, use the available range
            max_price = close_series.max()
            min_price = close_series.min()
            if max_price > min_price:  # Avoid division by zero
                df['Pct_From_52wk_High'] = (close_series / max_price - 1) * 100
                df['Pct_From_52wk_Low'] = (close_series / min_price - 1) * 100
            else:
                df['Pct_From_52wk_High'] = 0
                df['Pct_From_52wk_Low'] = 0
        
        # Breakout detection - adaptive to data length
        if data_length >= 5:
            # Use adaptive window based on available data
            breakout_window = min(12, max(3, data_length // 3))
            
            # Use n-period high as breakout reference
            rolling_high = high_series.rolling(window=breakout_window).max()
            rolling_low = low_series.rolling(window=breakout_window).min()
            
            # Detect price breakouts
            df['Price_Breakout'] = close_series > rolling_high.shift(1)
            
            # Check for volume confirmation if volume data is available
            if 'Vol_Ratio' in df and (df['Vol_Ratio'] > 0).any():
                df['Volume_Confirmed'] = df['Vol_Ratio'] > 1.2
                df['New_Breakout'] = df['Price_Breakout'] & df['Volume_Confirmed']
            else:
                # If no volume data, use price only
                df['New_Breakout'] = df['Price_Breakout']
            
            # Detect breakdown (bearish breakout)
            df['Price_Breakdown'] = close_series < rolling_low.shift(1)
            
            if 'Vol_Ratio' in df and (df['Vol_Ratio'] > 0).any():
                df['Breakdown'] = df['Price_Breakdown'] & df['Volume_Confirmed']
            else:
                df['Breakdown'] = df['Price_Breakdown']
        else:
            # For very limited data, just use placeholder values
            df['New_Breakout'] = False
            df['Breakdown'] = False
        
        # Consolidation pattern detection - adaptive to data length
        if data_length >= 5:
            # Adaptive window for consolidation detection
            window_size = min(8, max(3, data_length // 2))
            
            # Check if price is moving in a narrow range
            price_range = high_series.rolling(window=window_size).max() / low_series.rolling(window=window_size).min() - 1
            narrow_range = price_range < 0.05
            
            # Check for low volume if volume data is available
            if 'Vol_Ratio' in df and (df['Vol_Ratio'] > 0).any():
                low_volume = df['Vol_Ratio'] < 0.8
                df['Is_Consolidating'] = narrow_range & low_volume
            else:
                # If no volume data, use price range only
                df['Is_Consolidating'] = narrow_range
        else:
            df['Is_Consolidating'] = False
        
        # Price Range as % (High-Low)/Close
        df['Price_Range_Pct'] = (high_series - low_series) / close_series * 100
        
        logger.info(f"Calculated indicators with adaptive windows")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        traceback.print_exc()
        
        # Create a minimum viable dataframe to continue analysis
        if df is not None and 'Close' in df.columns:
            df['MA30'] = df['Close'].rolling(window=min(30, len(df))).mean()
            df['RSI'] = 50  # Neutral RSI
            logger.warning("Limited indicators calculated due to errors.")
        else:
            logger.error("Critical error: cannot calculate minimum viable indicators")
        
        return df

def identify_support_resistance(df, last_price=None):
    """Identify key support and resistance levels using local minima/maxima and volume analysis"""
    if df is None or len(df) < 30:
        return []
    
    try:
        # Extract high and low series safely
        high_series = get_safe_series(df, 'High')
        low_series = get_safe_series(df, 'Low')
        close_series = get_safe_series(df, 'Close')
        volume_series = get_safe_series(df, 'Volume')
        
        # Find local maxima and minima (rolling window of 5 periods)
        window = 5
        resistance_levels = []
        support_levels = []
        
        # Find resistance levels (local highs)
        for i in range(window, len(df) - window):
            if high_series.iloc[i] == high_series.iloc[i-window:i+window+1].max():
                # Check if volume was significant
                avg_vol = volume_series.iloc[i-window:i+window+1].mean()
                if volume_series.iloc[i] > avg_vol * 1.2:  # 20% above average
                    resistance_levels.append({
                        'price': float(high_series.iloc[i]),
                        'date': df.index[i],
                        'strength': 'strong' if volume_series.iloc[i] > avg_vol * 1.5 else 'medium'
                    })
                else:
                    resistance_levels.append({
                        'price': float(high_series.iloc[i]),
                        'date': df.index[i],
                        'strength': 'weak'
                    })
        
        # Find support levels (local lows)
        for i in range(window, len(df) - window):
            if low_series.iloc[i] == low_series.iloc[i-window:i+window+1].min():
                # Check if volume was significant
                avg_vol = volume_series.iloc[i-window:i+window+1].mean()
                if volume_series.iloc[i] > avg_vol * 1.2:  # 20% above average
                    support_levels.append({
                        'price': float(low_series.iloc[i]),
                        'date': df.index[i],
                        'strength': 'strong' if volume_series.iloc[i] > avg_vol * 1.5 else 'medium'
                    })
                else:
                    support_levels.append({
                        'price': float(low_series.iloc[i]),
                        'date': df.index[i],
                        'strength': 'weak'
                    })
        
        # Group nearby levels (within 3% of each other)
        def group_levels(levels):
            if not levels:
                return []
            
            # Sort by price
            sorted_levels = sorted(levels, key=lambda x: x['price'])
            
            # Group nearby levels
            grouped = []
            current_group = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                current_level = sorted_levels[i]
                prev_level = current_group[-1]
                
                # If current level is within 3% of previous level, add to current group
                if (current_level['price'] - prev_level['price']) / prev_level['price'] < 0.03:
                    current_group.append(current_level)
                else:
                    # Find average price weighted by strength
                    strength_weights = {'weak': 1, 'medium': 2, 'strong': 3}
                    total_weight = sum(strength_weights[level['strength']] for level in current_group)
                    avg_price = sum(level['price'] * strength_weights[level['strength']] for level in current_group) / total_weight
                    
                    # Determine overall strength
                    max_strength = max(level['strength'] for level in current_group)
                    
                    grouped.append({
                        'price': avg_price,
                        'date': max(level['date'] for level in current_group),
                        'strength': max_strength
                    })
                    
                    # Start new group
                    current_group = [current_level]
            
            # Add last group
            if current_group:
                strength_weights = {'weak': 1, 'medium': 2, 'strong': 3}
                total_weight = sum(strength_weights[level['strength']] for level in current_group)
                avg_price = sum(level['price'] * strength_weights[level['strength']] for level in current_group) / total_weight
                max_strength = max(level['strength'] for level in current_group)
                
                grouped.append({
                    'price': avg_price,
                    'date': max(level['date'] for level in current_group),
                    'strength': max_strength
                })
            
            return grouped
        
        # Group and combine levels
        resistance_levels = group_levels(resistance_levels)
        support_levels = group_levels(support_levels)
        
        # Add level type
        for level in resistance_levels:
            level['type'] = 'resistance'
        
        for level in support_levels:
            level['type'] = 'support'
        
        # Combine and sort by price
        all_levels = sorted(resistance_levels + support_levels, key=lambda x: x['price'])
        
        logger.info(f"Identified {len(resistance_levels)} resistance and {len(support_levels)} support levels")
        return all_levels
        
    except Exception as e:
        logger.error(f"Error identifying support/resistance levels: {str(e)}")
        traceback.print_exc()
        return []

def identify_weinstein_phase(df, current=None):
    """Identify the market phase according to Weinstein's method"""
    if df is None or len(df) < 4:  # Need at least 4 data points for basic phase analysis
        return 0, "Insufficient data"
        
    try:
        if current is None:
            current = df.iloc[-1]
        
        # Extract values safely and ensure we're working with scalar values
        def safe_get_value(row, column, default=0):
            if column not in row:
                return default
            
            value = row[column]
            
            # Handle different data types
            if isinstance(value, pd.Series):
                if value.empty:
                    return default
                if pd.isna(value.iloc[0]):
                    return default
                try:
                    return float(value.iloc[0])
                except:
                    return default
            elif isinstance(value, pd.DataFrame):
                if value.empty:
                    return default
                if pd.isna(value.iloc[0,0]):
                    return default
                try:
                    return float(value.iloc[0,0])
                except:
                    return default
            elif isinstance(value, bool):
                return value
            elif pd.isna(value):
                return default
            else:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
        
        # Safe extraction of key indicators
        close_value = safe_get_value(current, 'Close')
        
        # Check if we have MA30 (or closest available)
        if 'MA30' in current:
            ma30_value = safe_get_value(current, 'MA30')
        else:
            # If no MA30, try MA10 or just use a placeholder
            if 'MA10' in current:
                ma30_value = safe_get_value(current, 'MA10')
                logger.warning("Using MA10 as a substitute for MA30 due to limited data.")
            else:
                # If no MAs available, can't do proper phase analysis
                return 0, "Insufficient indicator data"
        
        # Get other indicators if available
        ma10_value = safe_get_value(current, 'MA10') if 'MA10' in current else 0
        ma30_slope_value = safe_get_value(current, 'MA30_Slope') if 'MA30_Slope' in current else 0
        ma30_slope_4wk_value = safe_get_value(current, 'MA30_Slope_4Wk') if 'MA30_Slope_4Wk' in current else 0
        rsi_value = safe_get_value(current, 'RSI') if 'RSI' in current else 50  # Default to neutral RSI
        
        # Extract boolean indicators (with defaults for missing data)
        is_consolidating = False
        if 'Is_Consolidating' in current:
            is_consolidating = bool(safe_get_value(current, 'Is_Consolidating', False))
        
        new_breakout = False
        if 'New_Breakout' in current:
            new_breakout = bool(safe_get_value(current, 'New_Breakout', False))
        
        breakdown = False
        if 'Breakdown' in current:
            breakdown = bool(safe_get_value(current, 'Breakdown', False))
        
        # Calculate trend conditions based on Weinstein's criteria
        price_above_ma30 = close_value > ma30_value
        ma10_above_ma30 = ma10_value > ma30_value
        ma30_slope_positive = ma30_slope_value > 0
        ma30_slope_improving = ma30_slope_value > ma30_slope_4wk_value / 4 if ma30_slope_4wk_value != 0 else False
        rsi_bullish = rsi_value > 50
        
        # Check recent price action - adaptive to available data
        min_periods = min(5, len(df))
        if min_periods >= 3:  # Need at least 3 periods to check for trend
            recent_df = df.iloc[-min_periods:]
            
            # Calculate higher highs and higher lows
            higher_highs = True
            higher_lows = True
            
            highs = get_safe_series(recent_df, 'High')
            lows = get_safe_series(recent_df, 'Low')
            
            if len(highs) >= 3 and not highs.isna().all():
                for i in range(1, len(highs)):
                    if highs.iloc[i] <= highs.iloc[i-1]:
                        higher_highs = False
                        break
            else:
                higher_highs = False
            
            if len(lows) >= 3 and not lows.isna().all():
                for i in range(1, len(lows)):
                    if lows.iloc[i] <= lows.iloc[i-1]:
                        higher_lows = False
                        break
            else:
                higher_lows = False
        else:
            higher_highs = False
            higher_lows = False
        
        # Phase identification with enhanced logic
        
        # Stage 2 (Uptrend) - Enhanced with strength indication and breakout detection
        if price_above_ma30 and ma30_slope_positive:
            if new_breakout:
                return 2, "Uptrend - New Breakout"
            elif higher_highs and higher_lows and ma30_slope_improving:
                return 2, "Strong Uptrend"
            elif ma10_above_ma30 and rsi_bullish:
                return 2, "Confirmed Uptrend"
            else:
                return 2, "Uptrend"
        
        # Stage 4 (Downtrend) - Enhanced with breakdown detection
        elif not price_above_ma30 and not ma30_slope_positive:
            if breakdown:
                return 4, "Strong Downtrend - New Breakdown"
            elif rsi_value < 30:
                return 4, "Downtrend - Oversold"
            else:
                return 4, "Downtrend"
        
        # Stage 3 (Top Formation)
        elif price_above_ma30 and not ma30_slope_positive:
            if rsi_value > 70:
                return 3, "Top Formation - Overbought"
            else:
                return 3, "Top Formation"
        
        # Stage 1 (Base Formation) - Enhanced with consolidation detection
        elif not price_above_ma30:
            if ma30_slope_positive or rsi_bullish:
                if is_consolidating and rsi_value > 45:
                    return 1, "Base Formation - Late Stage"
                else:
                    return 1, "Base Formation"
            else:
                return 1, "Base Formation - Early Stage"
        
        # Handle edge cases
        else:
            return 0, "Transition Phase"
            
    except Exception as e:
        logger.error(f"Error in phase identification: {str(e)}")
        traceback.print_exc()
        return 0, f"Error: {str(e)}"

def generate_recommendation(df, phase, phase_desc, market_context=None, sector_data=None):
    """Generate a trading recommendation based on Weinstein's method"""
    if phase == 0 or df is None or len(df) == 0:
        return "NO RECOMMENDATION - insufficient data"
        
    try:
        current = df.iloc[-1]
        
        # Safe extraction of values from potentially complex structures
        def safe_get_value(row, column, default=0):
            if column not in row:
                return default
                
            value = row[column]
            # Handle different data types
            if isinstance(value, pd.Series):
                if value.empty:
                    return default
                if pd.isna(value.iloc[0]):
                    return default
                try:
                    return float(value.iloc[0])
                except:
                    return default
            elif isinstance(value, pd.DataFrame):
                if value.empty:
                    return default
                if pd.isna(value.iloc[0,0]):
                    return default
                try:
                    return float(value.iloc[0,0])
                except:
                    return default
            elif isinstance(value, bool):
                return value
            elif pd.isna(value):
                return default
            else:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
        
        # Extract key indicators
        rsi = safe_get_value(current, 'RSI')
        vol_ratio = safe_get_value(current, 'Vol_Ratio')
        bb_width = safe_get_value(current, 'BB_Width')
        pct_from_52wk_high = safe_get_value(current, 'Pct_From_52wk_High')
        pct_from_ma30 = safe_get_value(current, 'Pct_From_MA30')
        
        # Extract boolean indicators
        new_breakout = False
        if 'New_Breakout' in current:
            new_breakout = bool(safe_get_value(current, 'New_Breakout', False))
        
        is_consolidating = False
        if 'Is_Consolidating' in current:
            is_consolidating = bool(safe_get_value(current, 'Is_Consolidating', False))
        
        # Consider market and sector context
        market_bullish = False
        sector_bullish = False
        
        if market_context is not None and market_context['phase'] in [1, 2]:
            market_bullish = True
        
        if sector_data is not None and sector_data['phase'] in [1, 2]:
            sector_bullish = True
        
        # Generate recommendation based on phase and indicators
        if phase == 2:  # Uptrend
            # Check for overbought conditions
            if rsi > 75 and pct_from_ma30 > 15:
                return "REDUCE POSITION / TIGHTEN STOPS - Overbought"
            # Check for strong uptrend with volume confirmation
            elif new_breakout and vol_ratio > 1.5 and market_bullish and sector_bullish:
                return "STRONG BUY - CONFIRMED BREAKOUT"
            elif new_breakout and vol_ratio > 1.2:
                return "BUY - BREAKOUT"
            # Check for healthy uptrend
            elif vol_ratio > 1.2 and 40 < rsi < 70 and market_bullish and sector_bullish:
                return "STRONG BUY - HEALTHY UPTREND"
            elif vol_ratio > 1.0 and 40 < rsi < 70:
                return "BUY"
            # Pullback opportunity
            elif -10 < pct_from_ma30 < -2 and rsi > 40:
                return "BUY - PULLBACK OPPORTUNITY"
            else:
                return "HOLD - UPTREND"
                
        elif phase == 1:  # Base Formation
            # Check for potential breakout setup
            if is_consolidating and rsi > 50 and vol_ratio > 0.8 and bb_width < 10:
                if market_bullish and sector_bullish:
                    return "ACCUMULATE - POTENTIAL BREAKOUT SOON"
                else:
                    return "WATCH CLOSELY - POTENTIAL BREAKOUT"
            # Check for early accumulation
            elif rsi > 45 and vol_ratio > 1.0:
                return "LIGHT ACCUMULATION - BASE BUILDING"
            # Early base
            elif is_consolidating:
                return "WATCH - BASE FORMING"
            else:
                return "MONITOR - WAIT FOR BASE COMPLETION"
            
        elif phase == 3:  # Top Formation
            # Check for distribution signs
            if rsi > 70:
                return "SELL / TAKE PROFITS - OVERBOUGHT"
            elif vol_ratio > 1.2 and pct_from_ma30 < 0:
                return "REDUCE POSITION - DISTRIBUTION SIGNS"
            else:
                return "TIGHTEN STOPS - TOP FORMING"
            
        elif phase == 4:  # Downtrend
            # Check for oversold bounce potential
            if rsi < 30 and vol_ratio > 1.5:
                return "AVOID / POTENTIAL OVERSOLD BOUNCE"
            # Strong downtrend
            elif vol_ratio > 1.2 and pct_from_52wk_high < -20:
                return "AVOID / SHORT OPPORTUNITY"
            else:
                return "AVOID - DOWNTREND"
        
        else:
            return "NEUTRAL - UNCLEAR PATTERN"
            
    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        traceback.print_exc()
        return "ERROR - No recommendation possible"

def generate_detailed_analysis(ticker_symbol, phase, phase_desc, recommendation, data, last_price=None, market_context=None, sector_data=None, support_resistance_levels=None):
    """Generate a detailed analysis text based on the identified phase and indicators"""
    if phase == 0 or data is None or len(data) == 0:
        return "Insufficient data for analysis."
    
    try:
        # Access the latest data point
        current = data.iloc[-1]
        
        # Safe extraction of values
        def safe_get_value(row, column, default=None):
            if column not in row:
                return default
            
            value = row[column]
            # Handle different data types
            if isinstance(value, pd.Series):
                if value.empty:
                    return default
                if pd.isna(value.iloc[0]):
                    return default
                try:
                    return float(value.iloc[0])
                except:
                    return default
            elif isinstance(value, pd.DataFrame):
                if value.empty:
                    return default
                if pd.isna(value.iloc[0,0]):
                    return default
                try:
                    return float(value.iloc[0,0])
                except:
                    return default
            elif isinstance(value, bool):
                return value
            elif pd.isna(value):
                return default
            else:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
        
        # Extract key indicators
        rsi = safe_get_value(current, 'RSI')
        vol_ratio = safe_get_value(current, 'Vol_Ratio')
        ma30_slope = safe_get_value(current, 'MA30_Slope')
        pct_from_ma30 = safe_get_value(current, 'Pct_From_MA30')
        pct_from_52wk_high = safe_get_value(current, 'Pct_From_52wk_High')
        
        # Market and sector context
        market_context_text = ""
        if market_context:
            market_phase_desc = {
                1: "Base Formation", 
                2: "Uptrend", 
                3: "Top Formation", 
                4: "Downtrend"
            }.get(market_context['phase'], "Unknown")
            
            market_context_text = f"The overall market (S&P 500) is in a {market_phase_desc} phase. "
            if market_context['phase'] == 2:
                market_context_text += "The market uptrend provides a favorable backdrop for bullish positions. "
            elif market_context['phase'] == 4:
                market_context_text += "The market downtrend suggests caution for long positions. "
        
        sector_context_text = ""
        if sector_data:
            sector_phase_desc = {
                1: "Base Formation", 
                2: "Uptrend", 
                3: "Top Formation", 
                4: "Downtrend"
            }.get(sector_data['phase'], "Unknown")
            
            sector_context_text = f"The {sector_data['name']} sector is in a {sector_phase_desc} phase. "
            if sector_data['relative_strength'] > 5:
                sector_context_text += f"This sector is showing strong relative strength (+{sector_data['relative_strength']:.1f}%) compared to the broader market. "
            elif sector_data['relative_strength'] < -5:
                sector_context_text += f"This sector is underperforming the broader market ({sector_data['relative_strength']:.1f}% relative strength). "
        
        # Find closest support and resistance levels
        levels_text = ""
        if support_resistance_levels:
            try:
                # Filter levels
                resistance_levels = [level for level in support_resistance_levels if level['type'] == 'resistance']
                support_levels = [level for level in support_resistance_levels if level['type'] == 'support']
                
                # Find nearest resistance level
                if resistance_levels and last_price:
                    nearest_resistance = min(resistance_levels, key=lambda x: abs(x['price'] - last_price))
                    resistance_price = float(nearest_resistance['price'])
                    resistance_pct = (resistance_price - last_price) / last_price * 100
                    
                    # Format with explicit direction symbol
                    if resistance_pct >= 0:
                        resistance_text = f"The nearest resistance is at ${resistance_price:.2f} (+{abs(resistance_pct):.1f}%). "
                    else:
                        resistance_text = f"The nearest resistance is at ${resistance_price:.2f} (-{abs(resistance_pct):.1f}%). "
                    
                    levels_text += resistance_text
                
                # Find nearest support level
                if support_levels and last_price:
                    nearest_support = min(support_levels, key=lambda x: abs(x['price'] - last_price))
                    support_price = float(nearest_support['price'])
                    support_pct = (last_price - support_price) / last_price * 100
                    
                    # Format with explicit direction symbol
                    if support_pct >= 0:
                        support_text = f"The nearest support is at ${support_price:.2f} (-{abs(support_pct):.1f}%)."
                    else:
                        support_text = f"The nearest support is at ${support_price:.2f} (+{abs(support_pct):.1f}%)."
                    
                    levels_text += support_text
            except Exception as e:
                logger.warning(f"Error formatting support/resistance levels: {str(e)}")
                levels_text = "Support and resistance levels could not be properly formatted."
        
        # Phase-specific analysis
        phase_analysis = ""
        if phase == 1:  # Base Formation
            phase_analysis = (
                f"{ticker_symbol} is in a Stage 1 Base Formation. "
                f"This is a consolidation phase following a downtrend where supply and demand are reaching equilibrium. "
            )
            
            if rsi and rsi > 50:
                phase_analysis += "RSI is above 50, indicating improving momentum within the base. "
            
            if ma30_slope and ma30_slope > 0:
                phase_analysis += "The 30-week moving average has begun to flatten and turn upward, a positive sign. "
            
            if vol_ratio and vol_ratio > 1.1:
                phase_analysis += "Volume is showing signs of accumulation, suggesting smart money may be taking positions. "
            
            phase_analysis += "According to Weinstein's method, the ideal time to buy is when the stock breaks out from this base into Stage 2 with increased volume. "
            
        elif phase == 2:  # Uptrend
            phase_analysis = (
                f"{ticker_symbol} is in a Stage 2 Uptrend. "
                f"This is the most profitable stage where price is trending higher with higher highs and higher lows. "
            )
            
            if pct_from_ma30 is not None:
                if pct_from_ma30 < -5:
                    phase_analysis += f"Price has pulled back {abs(pct_from_ma30):.1f}% from the 30-week MA, offering a potential buying opportunity. "
                elif pct_from_ma30 > 10:
                    phase_analysis += f"Price is extended {pct_from_ma30:.1f}% above its 30-week MA, suggesting caution and tighter stops. "
            
            if vol_ratio and vol_ratio > 1.2:
                phase_analysis += "Volume is confirming the uptrend with above-average participation. "
            
            if rsi:
                if rsi > 70:
                    phase_analysis += f"RSI is overbought at {rsi:.1f}, suggesting potential for a short-term pullback. "
                elif 40 < rsi < 70:
                    phase_analysis += f"RSI at {rsi:.1f} shows healthy momentum without being overbought. "
            
        elif phase == 3:  # Top Formation
            phase_analysis = (
                f"{ticker_symbol} is in a Stage 3 Top Formation. "
                f"This distribution phase typically occurs after a strong uptrend and precedes a downtrend. "
            )
            
            if ma30_slope and ma30_slope < 0:
                phase_analysis += "The 30-week moving average has started to roll over, a warning sign. "
            
            if vol_ratio and vol_ratio > 1.2:
                phase_analysis += "Higher volume on down days suggests distribution (selling) by institutions. "
            
            if rsi and rsi < 50:
                phase_analysis += "Declining RSI indicates weakening momentum. "
            
            phase_analysis += "According to Weinstein's method, this is typically a time to take profits or tighten stops rather than establishing new positions. "
            
        elif phase == 4:  # Downtrend
            phase_analysis = (
                f"{ticker_symbol} is in a Stage 4 Downtrend. "
                f"This bearish phase is characterized by lower highs and lower lows with price below a declining 30-week MA. "
            )
            
            if pct_from_52wk_high is not None:
                phase_analysis += f"Price is {abs(pct_from_52wk_high):.1f}% below its 52-week high. "
            
            if rsi and rsi < 30:
                phase_analysis += f"RSI is oversold at {rsi:.1f}, which may lead to short-term bounces, but the primary trend remains down. "
            
            phase_analysis += "According to Weinstein's method, Stage 4 stocks should be avoided for long positions. Wait for a Stage 1 base to form before considering entry. "
            
        # Combine all analysis components
        detailed_analysis = (
            f"{phase_analysis}\n\n"
            f"{market_context_text}{sector_context_text}\n\n"
            f"{levels_text}\n\n"
            f"Recommendation: {recommendation}"
        )
        
        logger.info(f"Generated detailed analysis for {ticker_symbol}")
        return detailed_analysis
        
    except Exception as e:
        logger.error(f"Error generating detailed analysis: {str(e)}")
        traceback.print_exc()
        return f"Error generating analysis: {str(e)}"
