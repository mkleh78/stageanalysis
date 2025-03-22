# Weinstein Ticker Analyzer

A comprehensive stock analysis tool based on Stan Weinstein's stage analysis methodology. This application helps investors identify market phases and make data-driven trading decisions.

## Features

- Stage Analysis: Automatically identifies the current Weinstein stage (Base, Uptrend, Top, Downtrend)
- Technical Indicators: Calculates and visualizes key indicators including:
  - Moving averages (10, 30, 50, 200-period)
  - RSI (Relative Strength Index)
  - Volume analysis
  - Bollinger Bands
  - Support/Resistance levels
- Interactive Charts: Comprehensive visualizations with candlestick patterns, indicators, and stage annotations
- Volume Profile: View volume distribution by price levels to identify significant support/resistance areas
- Sector Analysis: Compares individual ticker performance against its sector and the broader market
- Backtesting: Test the Weinstein strategy against historical data to evaluate performance

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
   git clone https://github.com/yourusername/weinstein-ticker-analyzer.git
   cd weinstein-ticker-analyzer

2. Install the required dependencies:
   pip install -r requirements.txt

### Dependencies

The project requires the following main packages:
- streamlit
- pandas
- numpy
- yfinance
- plotly
- datetime
- logging

A complete list can be found in the requirements.txt file.

## Usage

1. Start the Streamlit application:
   streamlit run app.py

2. Access the web interface in your browser (typically at http://localhost:8501)

3. Enter a ticker symbol (e.g., AAPL, MSFT, GOOGL)

4. Select the time period and interval for analysis

5. Click "Analyze" to run the Weinstein analysis

6. Navigate through the tabs to view different aspects of the analysis:
   - Overview: Summary of the analysis and key metrics
   - Chart: Interactive price chart with indicators
   - Support & Resistance: Key price levels
   - Volume Profile: Distribution of volume by price
   - Detailed Analysis: Text-based analysis of the current situation
   - Backtest: Performance simulation of the Weinstein strategy

## Sector ETF Files

For enhanced sector analysis, place sector ETF holding CSV files in the root directory of the application. The following files are supported:

- XLF.csv (Financials)
- XLK.csv (Information Technology)
- XLV.csv (Health Care)
- XLY.csv (Consumer Discretionary)
- XLP.csv (Consumer Staples)
- XLE.csv (Energy)
- XLB.csv (Materials)
- XLI.csv (Industrials)
- XLRE.csv (Real Estate)
- XLU.csv (Utilities)
- XLC.csv (Communication Services)

If these files are not available, the application will fall back to alternative methods for sector detection.

## Project Structure

weinstein_analyzer/
│
├── app.py                    # Main Streamlit application
├── weinstein_analyzer.py     # Main analyzer class
├── utils/
│   ├── __init__.py           # Makes the directory a package
│   ├── chart_utils.py        # Functions for charts and visualizations
│   ├── indicator_utils.py    # Technical indicators and analysis
│   ├── data_utils.py         # Data access and processing
│   └── backtest_utils.py     # Backtesting functionality
├── logging_config.py         # Logging configuration
├── requirements.txt          # Project dependencies
└── README.md                 # This file

## Weinstein's Stage Analysis

The application is based on Stan Weinstein's stage analysis methodology:

1. Stage 1 (Base Formation): Accumulation phase - The stock consolidates and forms a base. Look for signs of accumulation and improving relative strength.

2. Stage 2 (Uptrend): The most profitable stage - Price is trending higher with rising moving averages. This is the best time to buy and hold positions.

3. Stage 3 (Top Formation): Distribution phase - The uptrend loses momentum as early smart money begins to distribute shares. Time to take profits or tighten stops.

4. Stage 4 (Downtrend): Decline phase - Price moves lower with declining moving averages. Avoid long positions and wait for a Stage 1 base to form.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. It is not financial advice. Always do your own research before making investment decisions.
