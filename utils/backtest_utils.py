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
    
    Returns a dictionary with the backtest results.
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
        
        # Fixed parameters - very important, these are USD values
        initial_capital = 100000.0
        position_size_pct = 0.9  # 90% of capital used for position sizing
        stop_loss_pct = 0.07     # 7% stop loss
        
        logger.info(f"Starting simplified backtest for {ticker_symbol}")
        logger.info(f"Initial capital: ${initial_capital:.2f}")
        
        # Make sure we have clean price data
        clean_close = get_safe_series(df, 'Close').dropna()
        if len(clean_close) < 20:
            return {
                "success": False,
                "error": f"Not enough clean price data for backtest. Found only {len(clean_close)} valid price points."
            }
        
        # Prepare results DataFrame
        results = pd.DataFrame(index=df.index)
        results['Close'] = get_safe_series(df, 'Close')
        results['Signal'] = 'NONE'
        results['Phase'] = 0
        results['Position'] = 0  # 0: no position, 1: long
        results['Cash'] = float(initial_capital)  # Explicitly set as float
        results['Shares'] = 0.0
        results['Portfolio_Value'] = float(initial_capital)
        results['Trade_Start'] = False
        results['Trade_End'] = False
        results['Stop_Loss'] = 0.0
        
        # Variables for trading logic
        in_position = False
        entry_price = 0.0
        entry_date = None
        shares_held = 0.0
        stop_level = 0.0
        trades = []
        current_cash = float(initial_capital)
        
        # Main loop: Process each data point starting from the 30th day or earliest possible
        start_idx = min(30, len(df) - 10)  # Ensure we have at least 10 trading periods
        
        for i in range(start_idx, len(df)):
            # Historical data up to current day (exclusive)
            hist_window = df.iloc[:i]
            current_day = df.iloc[i]
            current_date = df.index[i]
            
            # Safely extract Close value
            try:
                # Ensure we have a floating point value
                current_price = float(current_day['Close'])
                if pd.isna(current_price):
                    logger.warning(f"NaN price at {current_date} - skipping")
                    continue  # Skip data with NaN values
            except Exception as e:
                logger.warning(f"Error accessing Close price at {current_date}: {str(e)}")
                continue
            
            # Get current Low price for stop-loss check (with fallback)
            try:
                day_low = float(get_safe_series(df, 'Low').iloc[i])
                if pd.isna(day_low):
                    day_low = current_price * 0.99  # Fallback: 1% below Close
            except Exception as e:
                logger.warning(f"Error accessing Low price at {current_date}: {str(e)}")
                day_low = current_price * 0.99  # Fallback
            
            # Identify phase and signal
            phase = _determine_historical_phase(hist_window, current_day)
            signal = _generate_signal_from_phase(phase, current_day)
            
            # Store phase and signal in results
            results.loc[current_date, 'Phase'] = phase
            results.loc[current_date, 'Signal'] = signal
            
            # Trading logic (entry and exit)
            if not in_position:  # Looking for entry
                # Check for buy signals
                if ('BUY' in signal or 'KAUFEN' in signal or 
                    'ACCUMULATE' in signal or 
                    (phase == 2 and 'WATCH' not in signal and 'NEUTRAL' not in signal)):  # Buy in Phase 2
                    
                    # Calculate position size based on current cash
                    position_value = current_cash * position_size_pct
                    
                    # Calculate number of shares to buy
                    if current_price > 0:  # Avoid division by zero
                        shares_to_buy = position_value / current_price
                        # Ensure shares_to_buy is a proper float
                        shares_to_buy = float(shares_to_buy)
                        
                        # Open position
                        in_position = True
                        entry_price = current_price
                        entry_date = current_date
                        shares_held = shares_to_buy
                        
                        # Update cash balance (deduct cost of shares)
                        current_cash -= (shares_held * current_price)
                        
                        # Set stop-loss (7% below entry price)
                        stop_level = entry_price * (1 - stop_loss_pct)
                        
                        # Update portfolio tracking dataframe
                        results.loc[current_date, 'Cash'] = current_cash
                        results.loc[current_date, 'Shares'] = shares_held
                        results.loc[current_date, 'Trade_Start'] = True
                        results.loc[current_date, 'Stop_Loss'] = stop_level
                        results.loc[current_date, 'Position'] = 1  # Now in position
                        
                        logger.info(f"BACKTEST: {current_date} - BUY {shares_held:.2f} shares at ${current_price:.2f}")
                        logger.info(f"Cash after purchase: ${current_cash:.2f}")
            
            else:  # Already in position - check for exit
                # Check for exit signals
                exit_signal = False
                exit_reason = ""
                exit_price = current_price
                
                # Stop-loss hit?
                if day_low <= stop_level:
                    exit_signal = True
                    exit_reason = "Stop-Loss"
                    exit_price = stop_level  # Assume sold at stop-loss price
                    logger.info(f"Stop-loss triggered at {current_date}: {stop_level:.2f}")
                
                # Sell signal from market phase?
                elif ('SELL' in signal or 'VERKAUFEN' in signal or 
                      'REDUCE' in signal or 'AVOID' in signal or
                      phase == 4):  # Always sell in Phase 4 (Downtrend)
                    exit_signal = True
                    exit_reason = "Sell Signal"
                
                # Close position if exit signal
                if exit_signal:
                    # Calculate profit/loss
                    profit_per_share = exit_price - entry_price
                    trade_profit_pct = (exit_price / entry_price - 1) * 100
                    trade_profit_usd = shares_held * profit_per_share
                    
                    # Update cash balance (add proceeds from sale)
                    current_cash += shares_held * exit_price
                    
                    # Update portfolio tracking dataframe
                    results.loc[current_date, 'Cash'] = current_cash
                    results.loc[current_date, 'Shares'] = 0.0
                    results.loc[current_date, 'Position'] = 0  # No longer in position
                    results.loc[current_date, 'Trade_End'] = True
                    
                    # Log trade
                    trade = {
                        'Entry_Date': entry_date,
                        'Entry_Price': float(entry_price),
                        'Exit_Date': current_date,
                        'Exit_Price': float(exit_price),
                        'Reason': exit_reason,
                        'Shares': float(shares_held),
                        'Profit_Percent': float(trade_profit_pct),
                        'Profit_USD': float(trade_profit_usd),
                        'Holding_Period': (current_date - entry_date).days
                    }
                    trades.append(trade)
                    
                    logger.info(f"BACKTEST: {current_date} - SELL {shares_held:.2f} shares at ${exit_price:.2f}")
                    logger.info(f"Trade profit: ${trade_profit_usd:.2f} ({trade_profit_pct:.2f}%)")
                    logger.info(f"Cash after sale: ${current_cash:.2f}")
                    
                    # Reset position variables
                    in_position = False
                    entry_price = 0.0
                    entry_date = None
                    shares_held = 0.0
                    stop_level = 0.0
                
                # Update trailing stop if price increases
                elif current_price > entry_price * 1.1:  # At least 10% profit
                    # Calculate new stop-loss (higher than previous)
                    new_stop = current_price * (1 - stop_loss_pct * 0.7)  # Tighter stop-loss
                    
                    # Only increase, never decrease
                    if new_stop > stop_level:
                        stop_level = new_stop
                        results.loc[current_date, 'Stop_Loss'] = stop_level
                        logger.info(f"Updated stop-loss at {current_date}: {stop_level:.2f}")
            
            # Calculate portfolio value (Cash + Position value)
            position_value = results.loc[current_date, 'Shares'] * current_price
            results.loc[current_date, 'Portfolio_Value'] = results.loc[current_date, 'Cash'] + position_value
        
        # Close last trade if still open (for meaningful comparison)
        if in_position:
            last_date = df.index[-1]
            last_price = float(df['Close'].iloc[-1])
            
            # Calculate profit/loss
            profit_per_share = last_price - entry_price
            trade_profit_pct = (last_price / entry_price - 1) * 100
            trade_profit_usd = shares_held * profit_per_share
            
            # Update cash balance
            current_cash += shares_held * last_price
            
            # Update portfolio
            results.loc[last_date, 'Cash'] = current_cash
            results.loc[last_date, 'Shares'] = 0.0
            results.loc[last_date, 'Position'] = 0
            results.loc[last_date, 'Trade_End'] = True
            results.loc[last_date, 'Portfolio_Value'] = current_cash  # Final portfolio value is all cash
            
            # Log trade
            trade = {
                'Entry_Date': entry_date,
                'Entry_Price': float(entry_price),
                'Exit_Date': last_date,
                'Exit_Price': float(last_price),
                'Reason': "End of test period",
                'Shares': float(shares_held),
                'Profit_Percent': float(trade_profit_pct),
                'Profit_USD': float(trade_profit_usd),
                'Holding_Period': (last_date - entry_date).days
            }
            trades.append(trade)
            
            logger.info(f"BACKTEST: {last_date} - FINAL TRADE CLOSED {shares_held:.2f} shares at ${last_price:.2f}")
            logger.info(f"Final trade profit: ${trade_profit_usd:.2f} ({trade_profit_pct:.2f}%)")
            logger.info(f"Final cash balance: ${current_cash:.2f}")
        
        # Check if trades occurred
        if len(trades) == 0:
            logger.warning("No trades generated in backtest period")
            return {
                "success": False,
                "error": "No trades were generated in the backtest period. Try a longer period or different stock."
            }
        
        # Calculate strategy return - TRIPLE CHECK APPROACH
        # Method 1: Using final portfolio value vs initial capital
        initial_portfolio_value = float(initial_capital)
        final_portfolio_value = float(results['Portfolio_Value'].iloc[-1])
        
        strategy_return_method1 = ((final_portfolio_value / initial_portfolio_value) - 1.0) * 100.0
        logger.info(f"Strategy return (Method 1): {strategy_return_method1:.2f}%")
        
        # Method 2: Sum all trade profits
        total_trade_profit_usd = sum(trade['Profit_USD'] for trade in trades)
        strategy_return_method2 = (total_trade_profit_usd / initial_portfolio_value) * 100.0
        logger.info(f"Strategy return (Method 2): {strategy_return_method2:.2f}%")
        
        # Method 3: Check portfolio value directly
        first_portfolio_value = float(results['Portfolio_Value'].iloc[0])
        last_portfolio_value = float(results['Portfolio_Value'].iloc[-1])
        strategy_return_method3 = ((last_portfolio_value / first_portfolio_value) - 1.0) * 100.0
        logger.info(f"Strategy return (Method 3): {strategy_return_method3:.2f}%")
        
        # Use Method 1 as our official return, but log discrepancies for debugging
        strategy_return = strategy_return_method1
        
        if abs(strategy_return_method1 - strategy_return_method2) > 0.1:
            logger.warning(f"Strategy return calculation discrepancy between methods 1 and 2: {abs(strategy_return_method1 - strategy_return_method2):.2f}%")
        
        if abs(strategy_return_method1 - strategy_return_method3) > 0.1:
            logger.warning(f"Strategy return calculation discrepancy between methods 1 and 3: {abs(strategy_return_method1 - strategy_return_method3):.2f}%")
            
        # Buy & Hold calculation - completely rewritten for reliability
        buy_hold_return = 0.0
        try:
            # Get clean price data for start and end
            valid_close = get_safe_series(df, 'Close').dropna()
            if len(valid_close) >= 2:
                first_price = float(valid_close.iloc[min(start_idx, len(valid_close)-1)])
                last_price = float(valid_close.iloc[-1])
                
                # Explicitly calculate buy and hold return
                if first_price > 0:
                    buy_hold_shares = initial_capital / first_price
                    buy_hold_end_value = buy_hold_shares * last_price
                    buy_hold_return = ((buy_hold_end_value / initial_capital) - 1.0) * 100.0
                    
                    logger.info(f"Buy & Hold: Bought {buy_hold_shares:.2f} shares at ${first_price:.2f}")
                    logger.info(f"Buy & Hold: End value ${buy_hold_end_value:.2f}")
                    logger.info(f"Buy & Hold return: {buy_hold_return:.2f}%")
        except Exception as e:
            logger.error(f"Error in Buy & Hold calculation: {str(e)}")
            buy_hold_return = 0.0
        
        # Calculate drawdown
        try:
            equity_series = results['Portfolio_Value']
            rolling_max = equity_series.expanding().max()
            drawdown = 100 * ((equity_series / rolling_max) - 1)
            max_drawdown = float(drawdown.min())
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            max_drawdown = 0.0
        
        # Trading statistics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['Profit_USD'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate trade summary (total P/L)
        total_profit_usd = sum(trade['Profit_USD'] for trade in trades if trade['Profit_USD'] > 0)
        total_loss_usd = sum(trade['Profit_USD'] for trade in trades if trade['Profit_USD'] <= 0)
        net_profit_usd = total_profit_usd + total_loss_usd
        
        # Average profit and loss
        avg_profit = 0.0
        avg_loss = 0.0
        
        if winning_trades > 0:
            avg_profit = sum(trade['Profit_Percent'] for trade in trades if trade['Profit_USD'] > 0) / winning_trades
        
        if losing_trades > 0:
            avg_loss = sum(trade['Profit_Percent'] for trade in trades if trade['Profit_USD'] <= 0) / losing_trades
        
        # Add trade summary to each trade for display
        for trade in trades:
            # Make sure all fields are proper Python types, not pandas or numpy types
            for key in trade:
                if isinstance(trade[key], (pd.Timestamp, np.datetime64)):
                    # Keep timestamps as is
                    pass
                elif isinstance(trade[key], (np.integer, np.floating)):
                    trade[key] = float(trade[key])
        
        # Complete results summary
        backtest_results = {
            "success": True,
            "ticker": ticker_symbol,
            "initial_capital": float(initial_capital),
            "final_equity": float(final_portfolio_value),
            "total_return_pct": float(strategy_return),
            "buy_hold_return_pct": float(buy_hold_return),
            "max_drawdown_pct": float(max_drawdown),
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "win_rate": float(win_rate),
            "avg_profit_pct": float(avg_profit),
            "avg_loss_pct": float(avg_loss),
            "total_profit_usd": float(total_profit_usd),
            "total_loss_usd": float(total_loss_usd),
            "net_profit_usd": float(net_profit_usd),
            "trades": trades,
            "equity_curve": results['Portfolio_Value'],
            "trade_signals": results[['Phase', 'Signal', 'Position', 'Trade_Start', 'Trade_End']]
        }
        
        logger.info(f"Backtest for {ticker_symbol} completed: {strategy_return:.2f}% total return")
        logger.info(f"Win rate: {win_rate*100:.1f}% ({winning_trades}/{total_trades} trades)")
        logger.info(f"Net profit: ${net_profit_usd:.2f}")
        return backtest_results
        
    except Exception as e:
        logger.error(f"Critical backtest error: {str(e)}")
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
    """Create chart for backtest results with position highlighting"""
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
        return fig
    
    try:
        # Prepare data from backtest results
        equity_curve = backtest_results['equity_curve']
        trade_signals = backtest_results['trade_signals']
        trades = backtest_results['trades']
        
        # Create equity curve chart with buy/hold comparison
        equity_fig = go.Figure()
        
        # Add position background shading
        # First, determine periods where we had an open position
        if 'Position' in trade_signals.columns:
            position_series = trade_signals['Position']
            
            # Get indices where position changes
            position_changes = position_series.diff().fillna(0) != 0
            change_points = position_series.index[position_changes].tolist()
            
            # Ensure we have start and end points
            if not change_points or change_points[0] != position_series.index[0]:
                change_points.insert(0, position_series.index[0])
            if change_points[-1] != position_series.index[-1]:
                change_points.append(position_series.index[-1])
            
            # Add background colors for positions
            for i in range(len(change_points) - 1):
                start_date = change_points[i]
                end_date = change_points[i+1]
                
                # Only color if we were in a position (Position == 1)
                position_value = position_series.loc[start_date]
                if position_value == 1:
                    equity_fig.add_shape(
                        type="rect",
                        x0=start_date,
                        x1=end_date,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        fillcolor="rgba(0,255,0,0.1)",  # Light green
                        line=dict(width=0),
                        layer="below"
                    )
        
        # Add equity curve line
        equity_fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            )
        )
        
        # Calculate and add Buy & Hold line
        if price_data is not None and len(price_data) > 0:
            try:
                initial_capital = backtest_results['initial_capital']
                
                # Get matching date range
                matching_dates = price_data.index.intersection(equity_curve.index)
                
                if len(matching_dates) > 1:
                    start_date = matching_dates[0]
                    start_price = float(price_data.loc[start_date])
                    
                    # Calculate shares bought with initial capital
                    buy_hold_shares = initial_capital / start_price
                    
                    # Calculate buy & hold portfolio value over time
                    buy_hold_values = price_data.loc[matching_dates] * buy_hold_shares
                    
                    # Add to chart
                    equity_fig.add_trace(
                        go.Scatter(
                            x=matching_dates,
                            y=buy_hold_values,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='gray', width=1.5, dash='dash')
                        )
                    )
            except Exception as e:
                logger.warning(f"Error adding Buy & Hold line: {str(e)}")
        
        # Add buy and sell markers
        buy_points = trade_signals[trade_signals['Trade_Start'] == True]
        sell_points = trade_signals[trade_signals['Trade_End'] == True]
        
        if not buy_points.empty:
            equity_fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=equity_curve.loc[buy_points.index],
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=12, color='green')
                )
            )
        
        if not sell_points.empty:
            equity_fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=equity_curve.loc[sell_points.index],
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=12, color='red')
                )
            )
        
        # Update equity chart layout
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
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Format y-axis with dollar signs
        equity_fig.update_yaxes(tickprefix="$")
        
        return equity_fig
        
    except Exception as e:
        logger.error(f"Error creating backtest chart: {str(e)}")
        traceback.print_exc()
        
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating backtest chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(height=600)
        return fig
