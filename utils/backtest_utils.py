import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import traceback
from utils.data_utils import get_safe_series

logger = logging.getLogger('WeinsteinAnalyzer')

def perform_simplified_backtest(df, ticker_symbol):
    """
    Simplified backtesting function for the Weinstein strategy.
    Uses fixed parameters: 100.000 USD initial capital and 90% position size.
    """
    if df is None or len(df) < 30:
        return {
            "success": False,
            "error": "Insufficient data for backtest. At least 30 data points required."
        }
    
    try:
        # Check if required indicators are present
        required_indicators = ['MA30', 'MA30_Slope', 'RSI']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        
        if missing_indicators:
            logger.warning(f"Missing indicators for backtest: {', '.join(missing_indicators)}")
            # Calculate missing indicators
            if 'MA30' not in df.columns:
                df['MA30'] = df['Close'].rolling(window=min(30, len(df))).mean()
                
            if 'MA30_Slope' not in df.columns:
                df['MA30_Slope'] = df['MA30'].diff()
                
            if 'RSI' not in df.columns:
                # Calculate RSI
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=min(14, len(df)//2)).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=min(14, len(df)//2)).mean()
                loss = loss.replace(0, np.nan)
                rs = gain / loss
                rs = rs.fillna(0)
                df['RSI'] = 100 - (100 / (1 + rs))
        
        # Fixed parameters
        initial_capital = 100000.0
        position_size_pct = 0.9
        stop_loss_pct = 0.07
        
        logger.info(f"Starting simplified backtest for {ticker_symbol}")
        
        # Prepare results DataFrame
        results = pd.DataFrame(index=df.index)
        results['Close'] = get_safe_series(df, 'Close')
        results['Signal'] = 'NONE'
        results['Phase'] = 0
        results['Position'] = 0  # 0: no position, 1: long
        results['Cash'] = initial_capital
        results['Shares'] = 0
        results['Portfolio_Value'] = initial_capital
        results['Trade_Start'] = False
        results['Trade_End'] = False
        results['Stop_Loss'] = 0.0
        
        # Variables for trading logic
        in_position = False
        entry_price = 0
        entry_date = None
        shares_held = 0
        stop_level = 0
        trades = []
        
        # Main loop: Process each data point starting from the 30th day or earliest possible
        start_idx = min(30, len(df) - 10)  # Ensure we have at least 10 trading periods
        
        for i in range(start_idx, len(df)):
            # Historical data up to current day (exclusive)
            hist_window = df.iloc[:i]
            current_day = df.iloc[i]
            current_date = df.index[i]
            
            # Safely extract Close value
            try:
                current_price = float(current_day['Close'])
                if pd.isna(current_price):
                    continue  # Skip data with NaN values
            except Exception as e:
                logger.warning(f"Error accessing Close price: {str(e)}")
                continue
            
            # Day's low for stop-loss check (with fallback)
            try:
                day_low = float(get_safe_series(df, 'Low').iloc[i])
                if pd.isna(day_low):
                    day_low = current_price * 0.99  # Fallback: 1% below Close
            except Exception as e:
                logger.warning(f"Error accessing Low price: {str(e)}")
                day_low = current_price * 0.99  # Fallback
            
            # Identify phase and signal
            phase = _determine_historical_phase(hist_window, current_day)
            signal = _generate_signal_from_phase(phase, current_day)
            
            # Store results
            results.loc[current_date, 'Phase'] = phase
            results.loc[current_date, 'Signal'] = signal
            
            # Trading logic (entry and exit)
            if not in_position:
                # Check for buy signals - with extended conditions
                if ('BUY' in signal or 'KAUFEN' in signal or 
                    'ACCUMULATE' in signal or 
                    (phase == 2 and 'WATCH' not in signal and 'NEUTRAL' not in signal)):  # Buy in Phase 2
                    
                    # Calculate position size
                    cash = results.loc[current_date, 'Cash']
                    position_value = cash * position_size_pct
                    shares_to_buy = position_value / current_price
                    
                    # Open position
                    in_position = True
                    entry_price = current_price
                    entry_date = current_date
                    shares_held = shares_to_buy
                    
                    # Set stop-loss (7% below entry price)
                    stop_level = entry_price * (1 - stop_loss_pct)
                    
                    # Update portfolio
                    results.loc[current_date, 'Cash'] = cash - (shares_to_buy * current_price)
                    results.loc[current_date, 'Shares'] = shares_to_buy
                    results.loc[current_date, 'Trade_Start'] = True
                    results.loc[current_date, 'Stop_Loss'] = stop_level
                    
                    logger.info(f"BACKTEST: {current_date} - BUY {shares_to_buy:.2f} shares at ${current_price:.2f}")
            
            else:  # In position
                # Check for exit signals
                exit_signal = False
                exit_reason = ""
                exit_price = current_price
                
                # Stop-loss hit?
                if day_low <= stop_level:
                    exit_signal = True
                    exit_reason = "Stop-Loss"
                    exit_price = stop_level  # Assume sell at stop-loss price
                
                # Sell signal?
                elif ('SELL' in signal or 'VERKAUFEN' in signal or 
                      'REDUCE' in signal or 'AVOID' in signal or
                      phase == 4):  # Always sell in Phase 4 (Downtrend)
                    exit_signal = True
                    exit_reason = "Sell Signal"
                
                # Close position if exit signal
                if exit_signal:
                    # Calculate profit/loss
                    trade_profit_pct = (exit_price / entry_price - 1) * 100
                    trade_profit = shares_held * (exit_price - entry_price)
                    
                    # Update portfolio
                    results.loc[current_date, 'Cash'] += shares_held * exit_price
                    results.loc[current_date, 'Shares'] = 0
                    results.loc[current_date, 'Trade_End'] = True
                    
                    # Log trade
                    trade = {
                        'Entry_Date': entry_date,
                        'Entry_Price': entry_price,
                        'Exit_Date': current_date,
                        'Exit_Price': exit_price,
                        'Reason': exit_reason,
                        'Shares': shares_held,
                        'Profit_Percent': trade_profit_pct,
                        'Profit_USD': trade_profit,
                        'Holding_Period': (current_date - entry_date).days
                    }
                    trades.append(trade)
                    
                    logger.info(f"BACKTEST: {current_date} - SELL {shares_held:.2f} shares at ${exit_price:.2f}, Profit: {trade_profit_pct:.2f}%")
                    
                    # Reset
                    in_position = False
                    entry_price = 0
                    entry_date = None
                    shares_held = 0
                    stop_level = 0
                
                # Update trailing stop if price increases
                elif current_price > entry_price * 1.1:  # At least 10% profit
                    # Calculate new stop-loss (higher than previous)
                    new_stop = current_price * (1 - stop_loss_pct * 0.7)  # Tighter stop-loss
                    
                    # Only increase, never decrease
                    if new_stop > stop_level:
                        stop_level = new_stop
                        results.loc[current_date, 'Stop_Loss'] = stop_level
        
            # Calculate portfolio value (Cash + Position value)
            position_value = results.loc[current_date, 'Shares'] * current_price
            results.loc[current_date, 'Portfolio_Value'] = results.loc[current_date, 'Cash'] + position_value
            results.loc[current_date, 'Position'] = 1 if in_position else 0
        
        # Close last trade if still open (for meaningful comparison)
        if in_position:
            last_date = df.index[-1]
            last_price = float(df['Close'].iloc[-1])
            
            # Calculate profit/loss
            trade_profit_pct = (last_price / entry_price - 1) * 100
            trade_profit = shares_held * (last_price - entry_price)
            
            # Update portfolio
            results.loc[last_date, 'Cash'] += shares_held * last_price
            results.loc[last_date, 'Shares'] = 0
            results.loc[last_date, 'Trade_End'] = True
            
            # Log trade
            trade = {
                'Entry_Date': entry_date,
                'Entry_Price': entry_price,
                'Exit_Date': last_date,
                'Exit_Price': last_price,
                'Reason': "End of test period",
                'Shares': shares_held,
                'Profit_Percent': trade_profit_pct,
                'Profit_USD': trade_profit,
                'Holding_Period': (last_date - entry_date).days
            }
            trades.append(trade)
            
            logger.info(f"BACKTEST: {last_date} - FINAL TRADE CLOSED {shares_held:.2f} shares at ${last_price:.2f}, Profit: {trade_profit_pct:.2f}%")
            
            # Update final portfolio value
            results.loc[last_date, 'Portfolio_Value'] = results.loc[last_date, 'Cash']
        
        # Check if trades occurred
        if len(trades) == 0:
            logger.warning("No trades generated in backtest period")
            return {
                "success": False,
                "error": "No trades were generated in the backtest period. Try a longer period or different stock."
            }
        
        # Calculate performance metrics
        # PROBLEM FIX: Ensure portfolio values are numeric and handle edge cases
        initial_portfolio_value = float(initial_capital)
        final_portfolio_value = float(results['Portfolio_Value'].iloc[-1])
        
        # Debug logging for troubleshooting
        logger.info(f"Initial portfolio value: {initial_portfolio_value}")
        logger.info(f"Final portfolio value: {final_portfolio_value}")
        
        # Calculate total return with robust approach
        if initial_portfolio_value > 0:
            total_return = ((final_portfolio_value / initial_portfolio_value) - 1.0) * 100
        else:
            total_return = 0
            logger.error("Initial portfolio value is zero or negative")
            
        logger.info(f"Calculated total return: {total_return}%")
            
        # Make Buy & Hold calculation more robust
        try:
            # Use first and last available price without NaN
            valid_close = get_safe_series(df, 'Close').dropna()
            if len(valid_close) >= 2:
                first_valid_close = float(valid_close.iloc[min(start_idx, len(valid_close)-1)])
                last_valid_close = float(valid_close.iloc[-1])
                if first_valid_close > 0:  # Prevent division by zero
                    buy_hold_return = (last_valid_close / first_valid_close - 1) * 100
                else:
                    buy_hold_return = 0
                    logger.warning("First valid close price is zero or negative")
            else:
                buy_hold_return = 0
                logger.warning("Insufficient valid data for Buy & Hold calculation")
        except Exception as e:
            logger.error(f"Error in Buy & Hold calculation: {str(e)}")
            buy_hold_return = 0
        
        # Calculate drawdown
        equity_series = results['Portfolio_Value']
        rolling_max = equity_series.expanding().max()
        drawdown = 100 * ((equity_series / rolling_max) - 1)
        max_drawdown = drawdown.min()
        
        # Trading statistics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['Profit_USD'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average profit and loss
        avg_profit = sum(trade['Profit_Percent'] for trade in trades if trade['Profit_USD'] > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(trade['Profit_Percent'] for trade in trades if trade['Profit_USD'] <= 0) / losing_trades if losing_trades > 0 else 0
        
        # Summarize results
        backtest_results = {
            "success": True,
            "ticker": ticker_symbol,
            "initial_capital": initial_portfolio_value,
            "final_equity": final_portfolio_value,
            "total_return_pct": total_return,
            "buy_hold_return_pct": buy_hold_return,
            "max_drawdown_pct": max_drawdown,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_profit_pct": avg_profit,
            "avg_loss_pct": avg_loss,
            "trades": trades,
            "equity_curve": results['Portfolio_Value'],
            "trade_signals": results[['Phase', 'Signal', 'Position', 'Trade_Start', 'Trade_End']]
        }
        
        logger.info(f"Backtest for {ticker_symbol} completed: {total_return:.2f}% total return")
        return backtest_results
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error during backtest: {str(e)}"
        }

def _determine_historical_phase(hist_data, current_day):
    """Improved function to determine Weinstein phase from historical data"""
    try:
        # Safe extraction of values with fallbacks
        try:
            close_value = float(current_day['Close'])
        except:
            # Try alternative access
            if isinstance(current_day['Close'], pd.Series):
                close_value = float(current_day['Close'].iloc[0])
            else:
                # Last resort
                close_value = float(hist_data['Close'].iloc[-1])
        
        # MA30 with fallback
        try:
            if 'MA30' in current_day:
                ma30_value = float(current_day['MA30'])
            elif 'MA10' in current_day:
                # Use MA10 as substitute
                ma30_value = float(current_day['MA10'])
            else:
                # Calculate temporary average
                ma30_window = min(30, len(hist_data))
                ma30_value = hist_data['Close'].tail(ma30_window).mean()
        except:
            # Fallback - MA30 as 95% of current price
            ma30_value = close_value * 0.95
            
        # MA30_Slope with fallback
        try:
            if 'MA30_Slope' in current_day:
                ma30_slope = float(current_day['MA30_Slope'])
            elif 'MA10_Slope' in current_day:
                ma30_slope = float(current_day['MA10_Slope'])
            else:
                # Calculate temporary simple slope
                if len(hist_data) >= 2:
                    last_close = hist_data['Close'].iloc[-1]
                    prev_close = hist_data['Close'].iloc[-2]
                    ma30_slope = last_close - prev_close
                else:
                    ma30_slope = 0
        except:
            # Fallback to neutral trend
            ma30_slope = 0
            
        # Weinstein phase determination
        price_above_ma30 = close_value > ma30_value
        ma30_slope_positive = ma30_slope > 0
        
        if price_above_ma30 and ma30_slope_positive:
            phase = 2  # Uptrend
        elif not price_above_ma30 and not ma30_slope_positive:
            phase = 4  # Downtrend
        elif price_above_ma30 and not ma30_slope_positive:
            phase = 3  # Top formation
        else:
            phase = 1  # Base formation
                
        return phase
            
    except Exception as e:
        logger.error(f"Error in historical phase identification: {str(e)}")
        # Fallback to Phase 1 (Base formation) as safe default
        return 1

def _generate_signal_from_phase(phase, current_day):
    """Improved function to generate signal with fallbacks"""
    try:
        # Extended signals based on phase
        if phase == 2:  # Uptrend
            # Check if we're in a strong uptrend
            try:
                rsi = float(current_day['RSI']) if 'RSI' in current_day else 60
                vol_ratio = float(current_day['Vol_Ratio']) if 'Vol_Ratio' in current_day else 1.0
                
                if rsi > 70:
                    return "SELL - Overbought"
                elif rsi > 60 and vol_ratio > 1.2:
                    return "BUY - Strong Uptrend"
                else:
                    return "BUY - Uptrend"
            except:
                return "BUY - Uptrend"
                
        elif phase == 1:  # Base formation
            # Try to use RSI for better signals
            try:
                if 'RSI' in current_day:
                    if isinstance(current_day['RSI'], pd.Series):
                        rsi = float(current_day['RSI'].iloc[0])
                    else:
                        rsi = float(current_day['RSI'])
                else:
                    rsi = 45
            except:
                rsi = 45
                
            if rsi > 50:
                return "ACCUMULATE - Late Base Formation"
            return "WATCH - Base Formation"
            
        elif phase == 3:  # Top formation
            return "SELL - Top Formation"
            
        elif phase == 4:  # Downtrend
            return "AVOID - Downtrend"
            
        else:
            # Unknown phase - neutral signal
            return "NEUTRAL - Unclear Phase"
            
    except Exception as e:
        logger.error(f"Error in signal generation: {str(e)}")
        # Fallback to neutral signal
        return "NEUTRAL - Error in signal generation"

def create_backtest_charts(backtest_results, price_data=None):
    """Create simplified charts for backtesting results with position highlighting"""
    if not backtest_results["success"]:
        # Create empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=backtest_results.get("error", "Backtest failed"),
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=600)
        return fig, fig
    
    try:
        # Prepare data from backtest results
        equity_curve = backtest_results['equity_curve']
        trade_signals = backtest_results['trade_signals']
        trades = backtest_results['trades']
        
        # 1. Equity curve with Buy & Hold comparison
        equity_fig = go.Figure()
        
        # IMPROVEMENT: Add position background shading
        # First, find continuous periods where position=1 (invested)
        if 'Position' in trade_signals.columns:
            position_changes = trade_signals['Position'].diff().fillna(0) != 0
            change_points = trade_signals.index[position_changes]
            
            if len(change_points) > 0:
                # Add the first and last data points to ensure complete coverage
                change_points = pd.Index([trade_signals.index[0]]).append(change_points)
                if change_points[-1] != trade_signals.index[-1]:
                    change_points = change_points.append(pd.Index([trade_signals.index[-1]]))
                
                # Create background shapes for invested periods
                for i in range(len(change_points) - 1):
                    start_idx = change_points[i]
                    end_idx = change_points[i+1]
                    
                    # Only color if we're invested
                    if trade_signals.loc[start_idx, 'Position'] == 1:
                        equity_fig.add_shape(
                            type="rect",
                            x0=start_idx,
                            x1=end_idx,
                            y0=0,  # Bottom of chart
                            y1=1,  # Top of chart
                            fillcolor="rgba(0,128,0,0.1)",  # Light green
                            opacity=0.5,
                            layer="below",
                            yref="paper",
                            line_width=0
                        )
        
        # Add equity curve
        equity_fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            )
        )
        
        # Buy & Hold line for comparison
        initial_capital = backtest_results['initial_capital']
        first_date = equity_curve.index[0]
        last_date = equity_curve.index[-1]
        
        if price_data is not None and len(price_data) >= len(equity_curve):
            price_subset = price_data.loc[first_date:last_date]
            if len(price_subset) > 0:
                # FIXED: Ensure proper scaling for Buy & Hold
                first_price = float(price_subset.iloc[0])
                if first_price > 0:  # Prevent division by zero
                    scale_factor = initial_capital / first_price
                    buy_hold_values = price_subset * scale_factor
                    
                    equity_fig.add_trace(
                        go.Scatter(
                            x=buy_hold_values.index,
                            y=buy_hold_values,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='gray', width=1.5, dash='dash')
                        )
                    )
        
        # Add buy and sell markers
        buy_dates = []
        buy_values = []
        sell_dates = []
        sell_values = []
        
        for i, row in trade_signals.iterrows():
            if row['Trade_Start']:
                buy_dates.append(i)
                buy_values.append(equity_curve.loc[i])
            elif row['Trade_End']:
                sell_dates.append(i)
                sell_values.append(equity_curve.loc[i])
        
        equity_fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_values,
                mode='markers',
                name='Buy',
                marker=dict(symbol='triangle-up', size=12, color='green')
            )
        )
        
        equity_fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_values,
                mode='markers',
                name='Sell',
                marker=dict(symbol='triangle-down', size=12, color='red')
            )
        )
        
        # Update layout
        equity_fig.update_layout(
            title=f"Equity Curve - {backtest_results['ticker']}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            yaxis=dict(
                tickprefix="$"
            )
        )
        
        # 2. Performance metrics table
        perf_fig = go.Figure()
        
        # Basic metrics
        metrics = {
            'Total Return': f"{backtest_results['total_return_pct']:.2f}%",
            'Buy & Hold Return': f"{backtest_results['buy_hold_return_pct']:.2f}%",
            'Excess Return': f"{backtest_results['total_return_pct'] - backtest_results['buy_hold_return_pct']:.2f}%",
            'Max Drawdown': f"{backtest_results['max_drawdown_pct']:.2f}%",
            'Number of Trades': str(backtest_results['total_trades']),
            'Winners': f"{backtest_results['winning_trades']} ({backtest_results['win_rate']*100:.1f}%)",
            'Losers': str(backtest_results['losing_trades']),
            'Avg. Win': f"{backtest_results['avg_profit_pct']:.2f}%",
            'Avg. Loss': f"{backtest_results['avg_loss_pct']:.2f}%"
        }
        
        # Create table
        perf_fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=14)
                ),
                cells=dict(
                    values=[
                        list(metrics.keys()),
                        list(metrics.values())
                    ],
                    fill_color=[['white', 'lightgrey'] * 5],
                    align='left',
                    font=dict(size=12)
                )
            )
        )
        
        perf_fig.update_layout(
            title="Performance Metrics",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return equity_fig, perf_fig
        
    except Exception as e:
        logger.error(f"Error creating backtest charts: {str(e)}")
        traceback.print_exc()
        
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating backtest charts: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(height=600)
        return fig, fig
