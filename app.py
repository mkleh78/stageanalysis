import streamlit as st
import pandas as pd
import logging
import warnings
import os

# Korrekte Import-Anweisung mit Fehlerbehandlung
try:
    from logging_config import setup_logging
except ModuleNotFoundError:
    # Fallback wenn das Modul nicht gefunden wird
    import logging
    import datetime

    def setup_logging():
        log_filename = f"weinstein_analyzer_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger('WeinsteinAnalyzer')
        return logger

# Import der WeinsteinTickerAnalyzer Klasse
from weinstein_analyzer import WeinsteinTickerAnalyzer

# Set up logging
logger = setup_logging()
warnings.filterwarnings('ignore')

def main():
    # Set page config
    st.set_page_config(
        page_title="Weinstein Ticker Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize the analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = WeinsteinTickerAnalyzer()
    
    # App title
    st.title("Weinstein Ticker Analyzer")
    st.subheader("Based on Stan Weinstein's Stage Analysis Method")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker Symbol", value="AAPL")
        
        # Use columns for a more compact layout
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox(
                "Period",
                options=["3mo", "6mo", "1y", "2y", "5y", "max"],
                index=2  # Default to 1y
            )
        with col2:
            interval = st.selectbox(
                "Interval",
                options=["1d", "1wk", "1mo"],
                index=1  # Default to 1wk
            )
        
        analyze_button = st.button(
            "Analyze",
            type="primary",
            use_container_width=True
        )
        
        # Info section in sidebar
        st.markdown("---")
        st.markdown("### Weinstein's Stages")
        st.markdown("""
        1. **Stage 1**: Base Formation - Accumulation phase
        2. **Stage 2**: Uptrend - Best time to buy
        3. **Stage 3**: Top Formation - Distribution phase 
        4. **Stage 4**: Downtrend - Avoid or consider shorting
        """)

    # Main content area
    if analyze_button:
        # Show a spinner while analyzing
        with st.spinner(f'Analyzing {ticker}...'):
            success = st.session_state.analyzer.load_data(ticker, period, interval)
        
        if success:
            # Create tabs for different analysis views
            tabs = st.tabs(["Overview", "Chart", "Support & Resistance", "Volume Profile", "Detailed Analysis", "Backtest"])
            
            # Get the analyzer instance from session state
            analyzer = st.session_state.analyzer
            
            # Display overview in first tab
            with tabs[0]:
                # Create status display with key information
                st.subheader("Analysis Summary")
                
                # Use columns for the status display
                col1, col2, col3 = st.columns(3)
                
                # Safe extraction of ticker name
                ticker_name = analyzer.ticker_info.get('longName', analyzer.ticker_info.get('shortName', analyzer.ticker_symbol))
                
                # Column 1: Ticker Info
                with col1:
                    st.markdown("##### Ticker Info")
                    st.markdown(f"**Symbol:** {analyzer.ticker_symbol}")
                    if ticker_name != analyzer.ticker_symbol:
                        st.markdown(f"**Name:** {ticker_name}")
                    st.markdown(f"**Last Price:** ${analyzer.last_price:.2f}" if analyzer.last_price is not None else "**Last Price:** N/A")
                
                # Column 2: Weinstein Analysis
                phase_colors = {
                    1: "#f0f0f0",  # Base - light gray
                    2: "#e6ffe6",  # Uptrend - light green
                    3: "#fff4e6",  # Top - light orange
                    4: "#ffe6e6"   # Downtrend - light red
                }
                phase_color = phase_colors.get(analyzer.phase, "#f9f9f9")
                
                with col2:
                    st.markdown("##### Weinstein Analysis")
                    st.markdown(f"**Phase:** {analyzer.phase} - {analyzer.phase_desc}")
                    st.markdown(f"**Recommendation:** {analyzer.recommendation}")
                
                # Column 3: Technical Indicators
                with col3:
                    st.markdown("##### Technical Indicators")
                    
                    # Safe extraction of RSI
                    rsi_value = "N/A"
                    if analyzer.data is not None and len(analyzer.data) > 0:
                        last_row = analyzer.data.iloc[-1]
                        if 'RSI' in last_row:
                            rsi_series = last_row['RSI']
                            if isinstance(rsi_series, pd.Series):
                                if not rsi_series.empty and not pd.isna(rsi_series.iloc[0]):
                                    rsi_value = f"{float(rsi_series.iloc[0]):.1f}"
                            elif not pd.isna(rsi_series):
                                rsi_value = f"{float(rsi_series):.1f}"
                    
                    # Safe extraction of Vol_Ratio
                    vol_ratio_value = "N/A"
                    if analyzer.data is not None and len(analyzer.data) > 0:
                        last_row = analyzer.data.iloc[-1]
                        if 'Vol_Ratio' in last_row:
                            vol_ratio_series = last_row['Vol_Ratio']
                            if isinstance(vol_ratio_series, pd.Series):
                                if not vol_ratio_series.empty and not pd.isna(vol_ratio_series.iloc[0]):
                                    vol_ratio_value = f"{float(vol_ratio_series.iloc[0]):.2f}x"
                            elif not pd.isna(vol_ratio_series):
                                vol_ratio_value = f"{float(vol_ratio_series):.2f}x"
                    
                    st.markdown(f"**RSI:** {rsi_value}")
                    st.markdown(f"**Volume Ratio:** {vol_ratio_value}")
                    
                # Market & Sector Context
                st.subheader("Market & Sector Context")
                
                # Use columns for market and sector info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Market Context")
                    if analyzer.market_context:
                        market_phase = {1: "Base", 2: "Uptrend", 3: "Top", 4: "Downtrend"}.get(
                            analyzer.market_context['phase'], "Unknown")
                        st.markdown(f"**Phase:** Stage {analyzer.market_context['phase']} ({market_phase})")
                        st.markdown(f"**S&P 500:** ${analyzer.market_context['last_close']:.2f}")
                        st.markdown(f"**1-Month Performance:** {analyzer.market_context['performance_1month']:.2f}%")
                    else:
                        st.markdown("No market context available")
                
                with col2:
                    st.markdown("##### Sector Context")
                    if analyzer.sector_data:
                        sector_phase = {1: "Base", 2: "Uptrend", 3: "Top", 4: "Downtrend"}.get(
                            analyzer.sector_data['phase'], "Unknown")
                        st.markdown(f"**Sector:** {analyzer.sector_data['name']} ({analyzer.sector_data['etf']})")
                        st.markdown(f"**Phase:** Stage {analyzer.sector_data['phase']} ({sector_phase})")
                        st.markdown(f"**Relative Strength:** {analyzer.sector_data['relative_strength']:.2f}%")
                    else:
                        st.markdown("No sector data available")
                
                # Display warnings if any
                if analyzer.warnings:
                    st.warning("Analysis Notes: " + " ".join(analyzer.warnings))
            
            # Display chart in second tab
            with tabs[1]:
                st.subheader("Price Chart with Weinstein Analysis")
                price_chart = analyzer.create_interactive_chart()
                st.plotly_chart(price_chart, use_container_width=True)
            
            # Display support and resistance levels in third tab
            with tabs[2]:
                st.subheader("Support & Resistance Levels")
                
                if analyzer.support_resistance_levels:
                    # Format levels for display
                    formatted_levels = []
                    for level in analyzer.support_resistance_levels:
                        # Calculate percentage distance from current price
                        if analyzer.last_price:
                            if level['type'] == 'resistance':
                                distance_pct = (level['price'] - analyzer.last_price) / analyzer.last_price * 100
                            else:  # support
                                distance_pct = (analyzer.last_price - level['price']) / analyzer.last_price * 100
                        else:
                            distance_pct = 0
                        
                        formatted_levels.append({
                            'Type': level['type'].capitalize(),
                            'Price': f"${level['price']:.2f}",
                            'Distance': f"{distance_pct:.1f}%",
                            'Strength': level['strength'].capitalize()
                        })
                    
                    # Create a dataframe for display
                    levels_df = pd.DataFrame(formatted_levels)
                    
                    # Apply conditional formatting
                    def highlight_level_type(row):
                        if row['Type'] == 'Support':
                            return ['background-color: rgba(0, 128, 0, 0.1); color: green' if col == 'Type' else '' for col in row.index]
                        elif row['Type'] == 'Resistance':
                            return ['background-color: rgba(255, 0, 0, 0.1); color: red' if col == 'Type' else '' for col in row.index]
                        return ['' for _ in row.index]
                    
                    def highlight_strength(row):
                        if row['Strength'] == 'Strong':
                            return ['font-weight: bold' if col == 'Strength' else '' for col in row.index]
                        return ['' for _ in row.index]
                    
                    # Display the table with styling
                    st.dataframe(
                        levels_df.style
                            .apply(highlight_level_type, axis=1)
                            .apply(highlight_strength, axis=1),
                        use_container_width=True
                    )
                else:
                    st.info("No support/resistance levels identified. This may be due to insufficient data or low volatility.")
            
            # Display volume profile in fourth tab
            with tabs[3]:
                st.subheader("Volume Profile Analysis")
                
                # Add a slider for lookback period
                lookback = st.slider(
                    "Lookback Period (number of periods)",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=10
                )
                
                volume_profile = analyzer.create_volume_profile(lookback_period=lookback)
                st.plotly_chart(volume_profile, use_container_width=True)
                
                st.markdown("""
                **About Volume Profile:**
                - Green bars represent volume at price levels above current price
                - Red bars represent volume at price levels below current price
                - The "Point of Control" (POC) is the price level with the highest volume
                - Significant volume nodes often act as support/resistance levels
                """)
            
            # Display detailed analysis in fifth tab
            with tabs[4]:
                st.subheader("Detailed Weinstein Analysis")
                
                # Format the detailed analysis with better styling
                detailed_text = analyzer.detailed_analysis
                if detailed_text:
                    # Split the analysis into sections
                    sections = detailed_text.split('\n\n')
                    
                    # Display each section with appropriate formatting
                    for i, section in enumerate(sections):
                        if i == 0:  # Phase analysis (first section)
                            st.markdown(f"### Phase Analysis\n{section}")
                        elif i == 1:  # Market and sector context
                            st.markdown(f"### Market & Sector Context\n{section}")
                        elif i == 2:  # Support and resistance levels
                            st.markdown(f"### Key Levels\n{section}")
                        elif i == 3:  # Recommendation
                            st.markdown(f"### {section}")
                else:
                    st.info("No detailed analysis available.")
            
            # Display backtest results in sixth tab
            with tabs[5]:
                st.subheader("Weinstein Strategy Backtest")
                
                # Comprehensible explanation of the backtest
                st.markdown("""
                The backtest simulates applying the Weinstein strategy to historical data with the following rules:
                - Initial capital: $100,000
                - Position size: 90% of available capital
                - Buy signals: When entering Phase 2 or on accumulation signals in late Phase 1
                - Sell signals: When entering Phase 3 or 4 or when stop-loss is hit
                - Stop-loss: 7% below entry price (raised if position is 10% in profit)
                """)
                
                # Show backtest results if available
                if analyzer.backtest_results and analyzer.backtest_results["success"]:
                    backtest_results = analyzer.backtest_results
                    
                    # Create backtest chart - note that it now returns only one chart
                    equity_chart = analyzer.create_simplified_backtest_charts(backtest_results)
                    
                    # Show strategy comparison
                    strategy_return = backtest_results["total_return_pct"]
                    buy_hold_return = backtest_results["buy_hold_return_pct"]
                    net_profit = backtest_results.get("net_profit_usd", 0.0)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Strategy Return",
                            f"{strategy_return:.2f}%",
                            f"{strategy_return - buy_hold_return:.2f}% vs. Buy & Hold"
                        )
                    
                    with col2:
                        st.metric(
                            "Buy & Hold Return",
                            f"{buy_hold_return:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Max. Drawdown",
                            f"{backtest_results['max_drawdown_pct']:.2f}%"
                        )
                    
                    # Show equity curve
                    st.subheader("Equity Curve and Trades")
                    st.plotly_chart(equity_chart, use_container_width=True)
                    
                    # Show trade history in collapsible section with total profit/loss summary
                    with st.expander("Trade History"):
                        # Display total profit/loss summary
                        total_profit = backtest_results.get("total_profit_usd", 0)
                        total_loss = backtest_results.get("total_loss_usd", 0)
                        net_profit = backtest_results.get("net_profit_usd", 0)
                        
                        profit_col, loss_col, net_col = st.columns(3)
                        with profit_col:
                            st.metric("Total Profit", f"${total_profit:.2f}", delta=None)
                        with loss_col:
                            st.metric("Total Loss", f"${total_loss:.2f}", delta=None)
                        with net_col:
                            st.metric("Net P/L", f"${net_profit:.2f}", delta=None)
                        
                        # Convert trade history to DataFrame for display
                        if backtest_results["trades"]:
                            trade_df = pd.DataFrame(backtest_results["trades"])
                            
                            # Format columns
                            trade_df["Entry_Date"] = pd.to_datetime(trade_df["Entry_Date"])
                            trade_df["Exit_Date"] = pd.to_datetime(trade_df["Exit_Date"])
                            
                            # Format price and profit columns
                            trade_df["Entry_Price"] = trade_df["Entry_Price"].map("${:.2f}".format)
                            trade_df["Exit_Price"] = trade_df["Exit_Price"].map("${:.2f}".format)
                            trade_df["Profit_Percent"] = trade_df["Profit_Percent"].map("{:.2f}%".format)
                            trade_df["Profit_USD"] = trade_df["Profit_USD"].map("${:.2f}".format)
                            
                            # Apply conditional formatting
                            def highlight_profit(row):
                                profit = row["Profit_Percent"]
                                profit_value = float(profit.replace("%", ""))
                                if profit_value > 0:
                                    return ["background-color: rgba(0,255,0,0.1)" if col == "Profit_Percent" else "" for col in row.index]
                                elif profit_value < 0:
                                    return ["background-color: rgba(255,0,0,0.1)" if col == "Profit_Percent" else "" for col in row.index]
                                return ["" for _ in row.index]
                            
                            # Sort by entry date
                            trade_df = trade_df.sort_values(by="Entry_Date")
                            
                            # Show formatted DataFrame
                            st.dataframe(
                                trade_df.style.apply(highlight_profit, axis=1),
                                use_container_width=True
                            )
                        else:
                            st.info("No trades executed during the backtest period.")
                    
                    # Brief explanation for interpretation
                    st.markdown("""
                    **Interpretation Notes:**
                    - **Excess Return**: Shows how much better or worse the strategy performed compared to a simple Buy & Hold strategy
                    - **Green triangles**: Buy signals according to the Weinstein strategy
                    - **Red triangles**: Sell signals (either due to phase change or stop-loss)
                    - **Green background**: Periods when you were invested (holding a position)
                    - **White background**: Periods without investment (cash position)
                    """)
                    
                else:
                    if analyzer.backtest_results:
                        # Backtest was performed but unsuccessful
                        st.error(f"Backtest failed: {analyzer.backtest_results.get('error', 'Unknown error')}")
                    else:
                        # No backtest available
                        st.info("No backtest available. Possibly insufficient data or an error occurred.")
                        
        else:
            # Show error message if analysis failed
            if st.session_state.analyzer.errors:
                st.error(f"Error analyzing {ticker}: {st.session_state.analyzer.errors[0]}")
            else:
                st.error(f"Failed to analyze {ticker}. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()
