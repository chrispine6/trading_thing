
# Mathematical & Statistical Market Analysis

Comprehensive suite of mathematical and statistical tools for market analysis and predictions.

## Overview

This package contains four main modules for analyzing financial market data:

1. **Price Predictions** - Forecasting future prices using multiple methods
2. **Trend Analysis** - Detecting and measuring market trends
3. **Volatility Predictions** - Forecasting volatility and risk metrics
4. **Pattern Recognition** - Identifying technical patterns and support/resistance

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Modules

### 1. Price Predictions (`price_predictions.py`)

Predicts future prices using multiple statistical methods:

- **Next period predictions**: Linear regression, ARIMA, Exponential Smoothing, Random Forest Ensemble
- **Price range forecasts**: Confidence intervals for price movements
- **Multi-step predictions**: Forecast multiple periods ahead
- **Date-specific predictions**: Predict price at a specific future date

**Usage:**
```python
from price_predictions import PricePredictor

predictor = PricePredictor('../../data/aapl.us.csv')

# Get all prediction methods
summary = predictor.get_prediction_summary()
print(summary)

# Price range with 95% confidence
ranges = predictor.predict_price_range(confidence=0.95)

# Multi-step forecast
forecast = predictor.predict_multi_step(steps=10)
```

### 2. Trend Analysis (`trend_analysis.py`)

Comprehensive trend detection and analysis:

- **Trend direction**: Uptrend, downtrend, or sideways (using linear regression, moving averages)
- **Trend strength**: ADX (Average Directional Index) with directional indicators
- **Momentum analysis**: Multi-timeframe momentum metrics and RSI
- **Reversal detection**: Identify potential trend reversal points
- **Cycle identification**: FFT and autocorrelation-based cycle detection

**Usage:**
```python
from trend_analysis import TrendAnalyzer

analyzer = TrendAnalyzer('../../data/aapl.us.csv')

# Detect trend direction
direction = analyzer.detect_trend_direction()
print(f"Current trend: {direction['trend']}")

# Calculate trend strength
strength = analyzer.calculate_trend_strength()
print(f"ADX: {strength['adx']:.2f}")

# Detect potential reversals
reversal = analyzer.detect_reversal_points()

# Identify market cycles
cycles = analyzer.detect_cycles()
```

### 3. Volatility Predictions (`volatility_predictions.py`)

Advanced volatility forecasting and risk analysis:

- **Historical volatility**: Multiple calculation methods (Parkinson, Garman-Klass)
- **GARCH forecasting**: Future volatility estimates using GARCH(1,1) model
- **Volatility clustering**: Detect periods of high/low volatility clustering
- **Breakout probability**: Predict likelihood of breakout from consolidation
- **Risk metrics**: Value at Risk (VaR) and Expected Shortfall (CVaR)

**Usage:**
```python
from volatility_predictions import VolatilityPredictor

predictor = VolatilityPredictor('../../data/aapl.us.csv')

# Historical volatility
hist_vol = predictor.estimate_historical_volatility()

# GARCH forecast
garch = predictor.predict_garch_volatility(horizon=5)

# Breakout analysis
breakout = predictor.calculate_breakout_probability()

# Risk metrics
var = predictor.calculate_var_es(confidence=0.95, horizon=1)
print(f"1-Day VaR: {var['var_historical_%']:.2f}%")
```

### 4. Pattern Recognition (`pattern_recognition.py`)

Technical pattern detection and analysis:

- **Similar patterns**: Find historical patterns similar to current price action
- **Candlestick patterns**: Detect Doji, Hammer, Engulfing, Stars, etc.
- **Chart patterns**: Triangles, Head & Shoulders, Double Tops/Bottoms
- **Support/Resistance**: Identify key price levels with strength metrics

**Usage:**
```python
from pattern_recognition import PatternRecognizer

recognizer = PatternRecognizer('../../data/aapl.us.csv')

# Find similar historical patterns
similar = recognizer.find_similar_patterns(window=20, top_n=5)

# Detect candlestick patterns
candles = recognizer.detect_candlestick_patterns()

# Identify chart patterns
charts = recognizer.detect_chart_patterns()

# Find support and resistance
sr = recognizer.identify_support_resistance()
```

## Running Complete Analysis

Use the orchestrator script to run all analyses at once:

```bash
# Run on default AAPL data
python run_all_analysis.py

# Run on specific file
python run_all_analysis.py ../../data/aat.us.csv

# Export results to JSON
python run_all_analysis.py ../../data/aapl.us.csv --export
```

## Individual Module Testing

Each module can be run independently:

```bash
python price_predictions.py
python volatility_predictions.py
python pattern_recognition.py
```

## Data Format

All modules expect CSV files with the following columns:
- `Date`: Date in YYYY-MM-DD format
- `Open`: Opening price
- `High`: Highest price
- `Low`: Lowest price
- `Close`: Closing price
- `Volume`: Trading volume
- `OpenInt`: Open interest (optional)

## Key Features

### Price Predictions
- ✓ Multiple prediction methods (Linear, ARIMA, Exponential Smoothing, Ensemble)
- ✓ Confidence intervals for predictions
- ✓ Multi-step ahead forecasting
- ✓ Technical indicators as features

### Trend Analysis
- ✓ ADX-based trend strength
- ✓ Multi-timeframe momentum
- ✓ RSI and MACD indicators
- ✓ FFT-based cycle detection
- ✓ Reversal probability

### Volatility Predictions
- ✓ GARCH(1,1) volatility forecasting
- ✓ Volatility clustering detection
- ✓ Breakout probability from consolidation
- ✓ VaR and Expected Shortfall
- ✓ Multiple volatility estimators

### Pattern Recognition
- ✓ Euclidean distance pattern matching
- ✓ 10+ candlestick patterns
- ✓ Chart pattern detection (triangles, H&S, etc.)
- ✓ Automated support/resistance identification
- ✓ Level strength metrics

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0
- scikit-learn >= 1.1.0
- statsmodels >= 0.13.0
- arch >= 5.3.0 (for GARCH models)

## Notes

- All modules handle missing data gracefully with fallback methods
- If ARCH package is not available, volatility module uses EWMA fallback
- Predictions are based on historical patterns and statistical models
- Always combine multiple indicators for better decision making

## Output Examples

### Price Predictions
```
Method      Target  Predicted_Price  Current_Price  Change_%
linear      Close   231.45          229.80         0.72
arima       Close   230.95          229.80         0.50
ensemble    Close   232.10          229.80         1.00
```

### Trend Analysis
```
Current Trend: Uptrend
Confidence: 100.0%
ADX: 28.45
Trend Strength: Strong
```

### Volatility Metrics
```
Annual Volatility: 24.56%
GARCH Forecast (5d): 25.12%
Volatility Regime: Moderate
1-Day VaR (95%): -1.48% ($3.42)
```

### Pattern Recognition
```
Candlestick Patterns: Bullish Engulfing (75% reliability)
Chart Patterns: Ascending Triangle (70% probability)
Support: $225.40 (-1.9%, Strength: 80)
Resistance: $235.20 (+2.4%, Strength: 75)
```

## License

For educational and research purposes.
python trend_analysis.py

