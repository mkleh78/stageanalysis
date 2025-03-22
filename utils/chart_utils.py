import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
import traceback
from utils.data_utils import get_safe_series

logger = logging.getLogger('WeinsteinAnalyzer')

def create_interactive_chart(data, ticker_symbol, phase, phase_desc, recommendation, support_resistance_levels=None, last_price=None):
    """Create an interactive chart with Plotly and enhanced visualization"""
    if data is None or len(data) == 0:
        # Create an empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=800)
        return fig
        
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price', 'Volume', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Safe extractions
        close = get_safe_series(data, 'Close')
        open_vals = get_safe_series(data, 'Open')
        high = get_safe_series(data, 'High')
        low = get_safe_series(data, 'Low')
        volume = get_safe_series(data, 'Volume')
        ma10 = get_safe_series(data, 'MA10')
        ma30 = get_safe_series(data, 'MA30')
        ma50 = get_safe_series(data, 'MA50')
        ma200 = get_safe_series(data, 'MA200')
        volma30 = get_safe_series(data, 'VolMA30')
        rsi = get_safe_series(data, 'RSI')
        bband_upper = get_safe_series(data, 'BBand_Upper')
        bband_lower = get_safe_series(data, 'BBand_Lower')
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=open_vals,
                high=high,
                low=low,
                close=close,
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma10,
                line=dict(color='blue', width=1.5),
                name="10-Week MA"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma30,
                line=dict(color='red', width=2),
                name="30-Week MA"
            ),
            row=1, col=1
        )
        
        if not ma50.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma50,
                    line=dict(color='green', width=1.5, dash='dot'),
                    name="50-Week MA"
                ),
                row=1, col=1
            )
        
        if not ma200.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma200,
                    line=dict(color='purple', width=1.5, dash='dash'),
                    name="200-Week MA"
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bband_upper,
                line=dict(color='rgba(0,0,0,0.3)', width=1),
                name="Upper BB",
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=bband_lower,
                line=dict(color='rgba(0,0,0,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0,0,0,0.05)',
                name="Lower BB",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add support and resistance levels
        if support_resistance_levels:
            for level in support_resistance_levels:
                color = 'green' if level['type'] == 'support' else 'red'
                width = 2 if level['strength'] == 'strong' else 1
                dash = 'solid' if level['strength'] == 'strong' else 'dash'
                
                fig.add_shape(
                    type="line",
                    x0=data.index[0],
                    y0=level['price'],
                    x1=data.index[-1],
                    y1=level['price'],
                    line=dict(
                        color=color,
                        width=width,
                        dash=dash
                    ),
                    row=1, col=1
                )
                
                # Add annotation only for strong levels
                if level['strength'] == 'strong':
                    fig.add_annotation(
                        x=data.index[-1],
                        y=level['price'],
                        text=f"{level['type'].capitalize()}: ${level['price']:.2f}",
                        showarrow=False,
                        xanchor="right",
                        yanchor="bottom" if level['type'] == 'resistance' else "top",
                        xshift=10,
                        font=dict(size=10, color=color)
                    )
        
        # Add volume bar chart with color-coding
        colors = []
        for i in range(len(data)):
            try:
                o = float(open_vals.iloc[i])
                c = float(close.iloc[i])
                colors.append('red' if o > c else 'green')
            except:
                colors.append('blue')  # Fallback color
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=volume,
                marker=dict(color=colors),
                name="Volume",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add volume moving average
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volma30,
                line=dict(color='orange', width=1.5),
                name="30-Week Vol MA",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi,
                line=dict(color='purple', width=1.5),
                name="RSI"
            ),
            row=3, col=1
        )
        
        # Add RSI reference lines
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[70, 70],
                line=dict(color='red', width=1, dash='dash'),
                name="Overbought",
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[30, 30],
                line=dict(color='green', width=1, dash='dash'),
                name="Oversold",
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[data.index[0], data.index[-1]],
                y=[50, 50],
                line=dict(color='gray', width=1, dash='dot'),
                name="Neutral",
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add background color based on the phase
        phase_colors = {
            1: 'rgba(128,128,128,0.15)',  # Base - gray
            2: 'rgba(0,128,0,0.1)',       # Uptrend - green
            3: 'rgba(255,165,0,0.1)',     # Top - orange
            4: 'rgba(255,0,0,0.1)'        # Downtrend - red
        }
        
        if phase in phase_colors:
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0,
                x1=1, y1=1,
                fillcolor=phase_colors[phase],
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        
        # Add Weinstein phase annotation
        phase_names = {
            1: "Stage 1: Base Formation",
            2: "Stage 2: Uptrend",
            3: "Stage 3: Top Formation",
            4: "Stage 4: Downtrend"
        }
        
        phase_name = phase_names.get(phase, "Unknown Phase")
        
        # Add breakout/breakdown annotations if detected
        if 'New_Breakout' in data.columns:
            breakout_points = data[data['New_Breakout'] == True]
            if not breakout_points.empty:
                for idx, point in breakout_points.iterrows():
                    try:
                        point_high = float(high.loc[idx])
                        fig.add_annotation(
                            x=idx,
                            y=point_high,
                            text="Breakout",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="green",
                            arrowsize=1,
                            arrowwidth=2,
                            ax=0,
                            ay=-40
                        )
                    except:
                        pass
        
        if 'Breakdown' in data.columns:
            breakdown_points = data[data['Breakdown'] == True]
            if not breakdown_points.empty:
                for idx, point in breakdown_points.iterrows():
                    try:
                        point_low = float(low.loc[idx])
                        fig.add_annotation(
                            x=idx,
                            y=point_low,
                            text="Breakdown",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="red",
                            arrowsize=1,
                            arrowwidth=2,
                            ax=0,
                            ay=40
                        )
                    except:
                        pass
        
        # Update layout with better styling
        fig.update_layout(
            title=f"{ticker_symbol}: {phase_name} - {phase_desc}<br><sub>{recommendation}</sub>",
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800,
            dragmode='zoom',
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=50, t=100, b=50),
            title_font=dict(size=16)
        )
        
        # Update RSI y-axis
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        
        # Update Volume y-axis
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive chart: {str(e)}")
        traceback.print_exc()
        
        # Create an empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(height=800)
        return fig

def create_volume_profile(data, ticker_symbol, last_price=None, lookback_period=None):
    """Create a volume profile for the specified timeframe"""
    if data is None or len(data) == 0:
        # Create an empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=600)
        return fig
        
    try:
        # Filter data based on lookback period if provided
        if lookback_period is not None and lookback_period < len(data):
            df = data.iloc[-lookback_period:].copy()
        else:
            df = data.copy()
            
        # Extract safe series for Close and Volume
        close_series = get_safe_series(df, 'Close')
        high_series = get_safe_series(df, 'High')
        low_series = get_safe_series(df, 'Low')
        volume_series = get_safe_series(df, 'Volume')
        
        # Check for empty data after cleaning
        if close_series.isna().all() or volume_series.isna().all():
            fig = go.Figure()
            fig.add_annotation(
                text="No valid data for volume profile",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
            
        # Remove NaN values
        valid_data = pd.DataFrame({
            'Close': close_series,
            'High': high_series,
            'Low': low_series,
            'Volume': volume_series
        }).dropna()
        
        if len(valid_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid data after removing NaN values",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
            
        # Use high-low range for better profile coverage
        min_price = float(valid_data['Low'].min())
        max_price = float(valid_data['High'].max())
        price_range = max_price - min_price
        
        if price_range <= 0:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient price data for volume profile",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
            
        # Determine optimal number of bins based on data size
        num_bins = min(max(20, len(valid_data) // 5), 50)
        if num_bins < 5:
            num_bins = 5  # Minimum 5 bins
            
        bin_size = price_range / num_bins
        
        # Calculate volume for each price bin with enhanced method
        volume_by_price = {}
        
        for i, row in valid_data.iterrows():
            bar_low = row['Low']
            bar_high = row['High']
            bar_volume = row['Volume']
            
            # Skip if invalid data
            if pd.isna(bar_low) or pd.isna(bar_high) or pd.isna(bar_volume) or bar_high <= bar_low:
                continue
            
            # Calculate which bins this bar spans
            low_bin = max(0, int((bar_low - min_price) / bin_size))
            high_bin = min(num_bins - 1, int((bar_high - min_price) / bin_size))
            
            # Evenly distribute volume across bins (could be weighted by time spent at each price)
            bins_spanned = max(1, high_bin - low_bin + 1)
            volume_per_bin = bar_volume / bins_spanned
            
            for bin_idx in range(low_bin, high_bin + 1):
                if bin_idx in volume_by_price:
                    volume_by_price[bin_idx] += volume_per_bin
                else:
                    volume_by_price[bin_idx] = volume_per_bin
        
        # Ensure all bins are represented
        all_bins = {i: volume_by_price.get(i, 0) for i in range(num_bins)}
        
        # Calculate bin midpoints for y-values
        y_values = [min_price + ((i + 0.5) * bin_size) for i in range(num_bins)]
        
        # Create volume profile chart
        fig = go.Figure()
        
        # Add volume bars (horizontal)
        fig.add_trace(
            go.Bar(
                x=list(all_bins.values()),
                y=y_values,
                orientation='h',
                marker=dict(
                    color=[
                        'rgba(0,128,0,0.7)' if y > last_price else 'rgba(255,0,0,0.7)' 
                        for y in y_values
                    ] if last_price is not None else 'rgba(0,0,255,0.7)'
                ),
                name="Volume Profile"
            )
        )
        
        # Add price line for current price
        if last_price is not None:
            max_vol = max(all_bins.values()) if all_bins and max(all_bins.values()) > 0 else 1
            
            fig.add_shape(
                type="line",
                x0=0,
                y0=last_price,
                x1=max_vol * 1.1,
                y1=last_price,
                line=dict(color="black", width=2, dash="dash"),
            )
            
            fig.add_annotation(
                x=max_vol * 1.05,
                y=last_price,
                text=f"Current: ${last_price:.2f}",
                showarrow=False,
                xanchor="right",
                font=dict(size=12)
            )
        
        # Add point of control (price level with highest volume)
        if all_bins:
            poc_bin = max(all_bins, key=all_bins.get)
            poc_price = y_values[poc_bin]
            poc_volume = all_bins[poc_bin]
            
            fig.add_shape(
                type="line",
                x0=0,
                y0=poc_price,
                x1=poc_volume,
                y1=poc_price,
                line=dict(color="blue", width=2),
            )
            
            fig.add_annotation(
                x=0,
                y=poc_price,
                text=f"POC: ${poc_price:.2f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="blue")
            )
        
        # Update layout
        lookback_text = f"past {lookback_period} periods" if lookback_period else "all data"
        
        fig.update_layout(
            title=f"Volume Profile for {ticker_symbol} ({lookback_text})",
            xaxis_title="Volume",
            yaxis_title="Price",
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating volume profile: {str(e)}")
        traceback.print_exc()
        
        # Create an empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating volume profile: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(height=600)
        return fig
