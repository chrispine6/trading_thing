"""
Trend Analysis Module
- Trend direction (uptrend, downtrend, sideways)
- Trend strength and momentum
- Trend reversal points
- Cycle identification (periodicity in price movements)
"""

import pandas as pd
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TrendAnalyzer:
    """Comprehensive trend analysis using statistical methods"""

    def __init__(self, data_path: str):
        """
        Initialize with OHLCV data

        Args:
            data_path: Path to CSV file with columns: Date, Open, High, Low, Close, Volume
        """
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.df.sort_index(inplace=True)

    def detect_trend_direction(self, window: int = 20) -> Dict[str, any]:
        """
        Detect current trend direction using multiple methods

        Args:
            window: Lookback window for trend detection

        Returns:
            Dictionary with trend information
        """
        close_prices = self.df['Close'].iloc[-window:]

        # Method 1: Linear regression slope
        x = np.arange(len(close_prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_prices)

        # Method 2: Moving average crossover
        sma_short = self.df['Close'].rolling(window=10).mean().iloc[-1]
        sma_long = self.df['Close'].rolling(window=window).mean().iloc[-1]

        # Method 3: Price position relative to moving averages
        current_price = self.df['Close'].iloc[-1]
        ma_20 = self.df['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = self.df['Close'].rolling(window=50).mean().iloc[-1] if len(self.df) >= 50 else ma_20

        # Determine trend
        trend_score = 0

        # Score based on slope
        if slope > 0:
            trend_score += 1
        elif slope < 0:
            trend_score -= 1

        # Score based on MA crossover
        if sma_short > sma_long:
            trend_score += 1
        elif sma_short < sma_long:
            trend_score -= 1

        # Score based on price position
        if current_price > ma_20 > ma_50:
            trend_score += 1
        elif current_price < ma_20 < ma_50:
            trend_score -= 1

        # Determine final trend
        if trend_score >= 2:
            trend = "Uptrend"
        elif trend_score <= -2:
            trend = "Downtrend"
        else:
            trend = "Sideways"

        return {
            'trend': trend,
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'confidence': abs(trend_score) / 3.0,
            'current_price': float(current_price),
            'ma_20': float(ma_20),
            'ma_50': float(ma_50),
            'price_change_%': float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100)
        }

    def calculate_trend_strength(self) -> Dict[str, float]:
        """
        Calculate trend strength using multiple indicators

        Returns:
            Dictionary with various strength metrics
        """
        close = self.df['Close']

        # ADX calculation (Average Directional Index)
        high = self.df['High']
        low = self.df['Low']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm_series = pd.Series(plus_dm, index=self.df.index)
        minus_dm_series = pd.Series(minus_dm, index=self.df.index)

        plus_di = 100 * (plus_dm_series.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm_series.rolling(window=14).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()

        # Momentum
        momentum = close - close.shift(10)

        # Rate of Change
        roc = close.pct_change(periods=10) * 100

        # Trend strength score (0-100)
        adx_value = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 20

        return {
            'adx': float(adx_value),
            'trend_strength': 'Strong' if adx_value > 25 else 'Weak',
            'plus_di': float(plus_di.iloc[-1]),
            'minus_di': float(minus_di.iloc[-1]),
            'momentum': float(momentum.iloc[-1]),
            'roc': float(roc.iloc[-1]),
            'interpretation': self._interpret_adx(adx_value)
        }

    def _interpret_adx(self, adx_value: float) -> str:
        """Interpret ADX value"""
        if adx_value < 20:
            return "Weak or no trend"
        elif adx_value < 25:
            return "Trend is starting to form"
        elif adx_value < 50:
            return "Strong trend"
        else:
            return "Very strong trend"

    def detect_momentum(self, windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate momentum across different timeframes

        Args:
            windows: List of lookback windows

        Returns:
            DataFrame with momentum metrics
        """
        if windows is None:
            windows = [5, 10, 20]
        close = self.df['Close']
        momentum_data = []

        for window in windows:
            # Price momentum
            price_momentum = close.iloc[-1] - close.iloc[-window]
            pct_change = (close.iloc[-1] / close.iloc[-window] - 1) * 100

            # Velocity (rate of change of momentum)
            if window >= 10:
                prev_momentum = close.iloc[-5] - close.iloc[-window-5]
                velocity = price_momentum - prev_momentum
            else:
                velocity = 0

            # RSI-like momentum
            gains = close.diff().clip(lower=0).rolling(window=window).mean()
            losses = -close.diff().clip(upper=0).rolling(window=window).mean()
            rs = gains / (losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            momentum_data.append({
                'Window': f'{window} periods',
                'Price_Momentum': float(price_momentum),
                'Percent_Change': float(pct_change),
                'Velocity': float(velocity),
                'RSI': float(rsi.iloc[-1]),
                'Signal': 'Bullish' if price_momentum > 0 else 'Bearish'
            })

        return pd.DataFrame(momentum_data)

    def detect_reversal_points(self, sensitivity: int = 5) -> Dict[str, any]:
        """
        Detect potential trend reversal points

        Args:
            sensitivity: Sensitivity of reversal detection (1-10, lower = more sensitive)

        Returns:
            Dictionary with reversal information
        """
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']

        # Calculate pivot points
        window = sensitivity * 2 + 1

        # Local maxima and minima
        recent_data = close.iloc[-100:]

        # Peak detection
        peaks, _ = signal.find_peaks(recent_data.values, distance=sensitivity)
        troughs, _ = signal.find_peaks(-recent_data.values, distance=sensitivity)

        # Check for recent divergence (price vs momentum)
        rsi = self._calculate_rsi(close, 14)
        macd, signal_line = self._calculate_macd(close)

        # Reversal signals
        reversal_signals = []

        # Overbought/Oversold
        current_rsi = rsi.iloc[-1]
        if current_rsi > 70:
            reversal_signals.append("Overbought (RSI > 70) - Potential bearish reversal")
        elif current_rsi < 30:
            reversal_signals.append("Oversold (RSI < 30) - Potential bullish reversal")

        # MACD crossover
        if len(macd) >= 2:
            if macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]:
                reversal_signals.append("MACD bullish crossover detected")
            elif macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1]:
                reversal_signals.append("MACD bearish crossover detected")

        # Price exhaustion
        recent_range = high.iloc[-20:].max() - low.iloc[-20:].min()
        current_volatility = close.iloc[-20:].std()

        if current_volatility > recent_range * 0.5:
            reversal_signals.append("High volatility - potential exhaustion")

        return {
            'reversal_probability': self._calculate_reversal_probability(reversal_signals, current_rsi),
            'signals': reversal_signals,
            'rsi': float(current_rsi),
            'recent_peaks': len(peaks),
            'recent_troughs': len(troughs),
            'last_peak_distance': int(100 - peaks[-1]) if len(peaks) > 0 else None,
            'last_trough_distance': int(100 - troughs[-1]) if len(troughs) > 0 else None
        }

    def _calculate_reversal_probability(self, signals: List[str], rsi: float) -> str:
        """Calculate reversal probability"""
        signal_count = len(signals)

        if signal_count == 0:
            return "Low (0-25%)"
        elif signal_count == 1:
            return "Moderate (25-50%)"
        elif signal_count == 2:
            return "High (50-75%)"
        else:
            return "Very High (75-100%)"

    def detect_cycles(self, min_period: int = 5, max_period: int = 50) -> Dict[str, any]:
        """
        Identify cycles and periodicity in price movements using FFT

        Args:
            min_period: Minimum cycle period to detect
            max_period: Maximum cycle period to detect

        Returns:
            Dictionary with cycle information
        """
        close = self.df['Close'].iloc[-500:] if len(self.df) >= 500 else self.df['Close']

        # Detrend the data
        detrended = signal.detrend(close.values)

        # Apply FFT
        n = len(detrended)
        yf = fft(detrended)
        xf = fftfreq(n, 1)[:n//2]

        # Get power spectrum
        power = 2.0/n * np.abs(yf[0:n//2])

        # Find dominant frequencies
        valid_indices = np.where((xf > 1/max_period) & (xf < 1/min_period))[0]

        if len(valid_indices) > 0:
            valid_power = power[valid_indices]
            valid_freq = xf[valid_indices]

            # Get top 3 cycles
            top_indices = np.argsort(valid_power)[-3:][::-1]

            cycles = []
            for idx in top_indices:
                period = 1 / valid_freq[idx] if valid_freq[idx] != 0 else 0
                cycles.append({
                    'period_days': float(period),
                    'strength': float(valid_power[idx]),
                    'frequency': float(valid_freq[idx])
                })
        else:
            cycles = []

        # Alternative: Autocorrelation method
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find peaks in autocorrelation
        peaks, properties = signal.find_peaks(autocorr[1:50], height=0.3, distance=5)

        autocorr_cycles = []
        if len(peaks) > 0:
            for peak in peaks[:3]:
                autocorr_cycles.append({
                    'period_days': int(peak + 1),
                    'correlation': float(autocorr[peak + 1])
                })

        return {
            'fft_cycles': cycles,
            'autocorrelation_cycles': autocorr_cycles,
            'dominant_cycle': cycles[0]['period_days'] if cycles else None,
            'cycle_strength': cycles[0]['strength'] if cycles else 0
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd, signal

    def get_comprehensive_trend_report(self) -> Dict[str, any]:
        """Generate comprehensive trend analysis report"""
        return {
            'trend_direction': self.detect_trend_direction(),
            'trend_strength': self.calculate_trend_strength(),
            'momentum': self.detect_momentum().to_dict('records'),
            'reversal_analysis': self.detect_reversal_points(),
            'cycle_analysis': self.detect_cycles()
        }


if __name__ == "__main__":
    # Example usage
    analyzer = TrendAnalyzer('../../data/aapl.us.csv')

    print("=" * 80)
    print("TREND DIRECTION ANALYSIS")
    print("=" * 80)
    direction = analyzer.detect_trend_direction()
    print(f"\nCurrent Trend: {direction['trend']}")
    print(f"Confidence: {direction['confidence']:.1%}")
    print(f"R-Squared: {direction['r_squared']:.4f}")
    print(f"Price Change: {direction['price_change_%']:.2f}%")
    print(f"Current Price: ${direction['current_price']:.2f}")
    print(f"MA(20): ${direction['ma_20']:.2f}")
    print(f"MA(50): ${direction['ma_50']:.2f}")

    print("\n" + "=" * 80)
    print("TREND STRENGTH")
    print("=" * 80)
    strength = analyzer.calculate_trend_strength()
    print(f"\nADX: {strength['adx']:.2f}")
    print(f"Trend Strength: {strength['trend_strength']}")
    print(f"Interpretation: {strength['interpretation']}")
    print(f"+DI: {strength['plus_di']:.2f}")
    print(f"-DI: {strength['minus_di']:.2f}")
    print(f"Momentum: {strength['momentum']:.2f}")
    print(f"ROC: {strength['roc']:.2f}%")

    print("\n" + "=" * 80)
    print("MOMENTUM ANALYSIS")
    print("=" * 80)
    momentum = analyzer.detect_momentum()
    print("\n", momentum.to_string(index=False))

    print("\n" + "=" * 80)
    print("REVERSAL POINT DETECTION")
    print("=" * 80)
    reversal = analyzer.detect_reversal_points()
    print(f"\nReversal Probability: {reversal['reversal_probability']}")
    print(f"RSI: {reversal['rsi']:.2f}")
    print(f"Recent Peaks: {reversal['recent_peaks']}")
    print(f"Recent Troughs: {reversal['recent_troughs']}")
    print("\nSignals:")
    for signal in reversal['signals']:
        print(f"  • {signal}")

    print("\n" + "=" * 80)
    print("CYCLE IDENTIFICATION")
    print("=" * 80)
    cycles = analyzer.detect_cycles()
    print("\nFFT-Based Cycles:")
    for i, cycle in enumerate(cycles['fft_cycles'], 1):
        print(f"  {i}. Period: {cycle['period_days']:.1f} days, Strength: {cycle['strength']:.4f}")

    print("\nAutocorrelation-Based Cycles:")
    for i, cycle in enumerate(cycles['autocorrelation_cycles'], 1):
        print(f"  {i}. Period: {cycle['period_days']} days, Correlation: {cycle['correlation']:.4f}")
"""
Price Predictions Module
- Next period's price (close, high, low)
- Price range forecasts
- Multi-step ahead price predictions
- Expected price at specific future dates
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PricePredictor:
    """Comprehensive price prediction using multiple statistical methods"""

    def __init__(self, data_path: str):
        """
        Initialize with OHLCV data

        Args:
            data_path: Path to CSV file with columns: Date, Open, High, Low, Close, Volume
        """
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.df.sort_index(inplace=True)

    def add_features(self, lookback: int = 20) -> pd.DataFrame:
        """Add technical features for prediction"""
        df = self.df.copy()

        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()

        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(lookback)
        df['ROC'] = df['Close'].pct_change(periods=lookback)

        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # Price range
        df['High_Low_Range'] = df['High'] - df['Low']
        df['Close_Open_Range'] = df['Close'] - df['Open']

        # Lagged features
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)

        return df.dropna()

    def predict_next_price(self, method: str = 'all') -> Dict[str, float]:
        """
        Predict next period's close, high, low prices

        Args:
            method: 'linear', 'arima', 'exponential', 'ensemble', or 'all'

        Returns:
            Dictionary with predicted prices
        """
        results = {}

        if method in ['linear', 'all']:
            results['linear'] = self._linear_regression_prediction()

        if method in ['arima', 'all']:
            results['arima'] = self._arima_prediction()

        if method in ['exponential', 'all']:
            results['exponential'] = self._exponential_smoothing_prediction()

        if method in ['ensemble', 'all']:
            results['ensemble'] = self._ensemble_prediction()

        return results

    def _linear_regression_prediction(self) -> Dict[str, float]:
        """Linear regression based prediction"""
        df = self.add_features()

        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']]

        # Train on last 252 days (1 trading year)
        train_data = df.iloc[-252:]
        X_train = train_data[feature_cols].values

        # Predict Close, High, Low
        predictions = {}

        for target in ['Close', 'High', 'Low']:
            y_train = train_data[target].values
            model = LinearRegression()
            model.fit(X_train[:-1], y_train[1:])  # Predict next day

            # Predict next value
            X_last = X_train[-1:].reshape(1, -1)
            predictions[target.lower()] = float(model.predict(X_last)[0])

        return predictions

    def _arima_prediction(self) -> Dict[str, float]:
        """ARIMA model prediction"""
        predictions = {}

        for target in ['Close', 'High', 'Low']:
            try:
                # Use last 252 days
                series = self.df[target].iloc[-252:]

                # Fit ARIMA(1,1,1) - simple configuration
                model = ARIMA(series, order=(1, 1, 1))
                fitted = model.fit()

                # Forecast next period
                forecast = fitted.forecast(steps=1)
                predictions[target.lower()] = float(forecast.iloc[0])
            except:
                # Fallback to simple moving average
                predictions[target.lower()] = float(self.df[target].iloc[-5:].mean())

        return predictions

    def _exponential_smoothing_prediction(self) -> Dict[str, float]:
        """Exponential smoothing prediction"""
        predictions = {}

        for target in ['Close', 'High', 'Low']:
            try:
                series = self.df[target].iloc[-252:]

                # Fit Holt-Winters model
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=None,
                    initialization_method='estimated'
                )
                fitted = model.fit()
                forecast = fitted.forecast(steps=1)
                predictions[target.lower()] = float(forecast.iloc[0])
            except:
                # Fallback to exponential weighted mean
                predictions[target.lower()] = float(series.ewm(span=20).mean().iloc[-1])

        return predictions

    def _ensemble_prediction(self) -> Dict[str, float]:
        """Ensemble prediction combining multiple methods"""
        df = self.add_features()

        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']]
        train_data = df.iloc[-252:]
        X_train = train_data[feature_cols].values

        predictions = {}

        for target in ['Close', 'High', 'Low']:
            y_train = train_data[target].values

            # Random Forest for ensemble
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train[:-1], y_train[1:])

            X_last = X_train[-1:].reshape(1, -1)
            predictions[target.lower()] = float(model.predict(X_last)[0])

        return predictions

    def predict_price_range(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Predict price range with confidence intervals

        Args:
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dictionary with (lower_bound, upper_bound) for each price type
        """
        df = self.add_features()
        predictions = {}

        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

        for target in ['Close', 'High', 'Low']:
            # Calculate historical volatility
            returns = df[target].pct_change().dropna()
            std = returns.std()

            # Get current price and predicted price
            current_price = self.df[target].iloc[-1]

            # Simple prediction using EMA
            predicted = df[target].ewm(span=20).mean().iloc[-1]

            # Calculate confidence interval
            lower_bound = predicted * (1 - z_score * std)
            upper_bound = predicted * (1 + z_score * std)

            predictions[target.lower()] = (float(lower_bound), float(upper_bound))

        return predictions

    def predict_multi_step(self, steps: int = 5, method: str = 'arima') -> pd.DataFrame:
        """
        Multi-step ahead predictions

        Args:
            steps: Number of periods to predict
            method: Prediction method to use

        Returns:
            DataFrame with predictions for each step
        """
        predictions = []

        for target in ['Close', 'High', 'Low']:
            if method == 'arima':
                try:
                    series = self.df[target].iloc[-252:]
                    model = ARIMA(series, order=(1, 1, 1))
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=steps)

                    for i, value in enumerate(forecast):
                        predictions.append({
                            'Step': i + 1,
                            'Target': target,
                            'Predicted_Price': float(value)
                        })
                except:
                    # Fallback: use trend extrapolation
                    last_price = self.df[target].iloc[-1]
                    trend = (self.df[target].iloc[-1] - self.df[target].iloc[-21]) / 20

                    for i in range(steps):
                        predictions.append({
                            'Step': i + 1,
                            'Target': target,
                            'Predicted_Price': float(last_price + trend * (i + 1))
                        })

        return pd.DataFrame(predictions)

    def predict_at_date(self, target_date: str, method: str = 'arima') -> Dict[str, float]:
        """
        Predict expected price at a specific future date

        Args:
            target_date: Target date as string (YYYY-MM-DD)
            method: Prediction method

        Returns:
            Dictionary with predicted prices
        """
        target = pd.to_datetime(target_date)
        last_date = self.df.index[-1]

        # Calculate trading days (approximately)
        days_diff = (target - last_date).days
        steps = int(days_diff * 5 / 7)  # Approximate trading days

        if steps <= 0:
            return {"error": "Target date must be in the future"}

        multi_step = self.predict_multi_step(steps=steps, method=method)

        # Get final predictions
        final_predictions = multi_step[multi_step['Step'] == steps]

        result = {}
        for _, row in final_predictions.iterrows():
            result[row['Target'].lower()] = row['Predicted_Price']

        return result

    def get_prediction_summary(self) -> pd.DataFrame:
        """Get comprehensive summary of all prediction methods"""
        all_predictions = self.predict_next_price(method='all')

        summary_data = []
        for method, preds in all_predictions.items():
            for target, value in preds.items():
                summary_data.append({
                    'Method': method,
                    'Target': target.capitalize(),
                    'Predicted_Price': value,
                    'Current_Price': float(self.df[target.capitalize()].iloc[-1]),
                    'Change_%': ((value / float(self.df[target.capitalize()].iloc[-1])) - 1) * 100
                })

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    predictor = PricePredictor('../../data/aapl.us.csv')

    print("=" * 80)
    print("PRICE PREDICTIONS FOR NEXT PERIOD")
    print("=" * 80)

    # Get all predictions
    summary = predictor.get_prediction_summary()
    print("\nPrediction Summary:")
    print(summary.to_string(index=False))

    # Price range forecast
    print("\n" + "=" * 80)
    print("PRICE RANGE FORECASTS (95% Confidence)")
    print("=" * 80)
    ranges = predictor.predict_price_range()
    for target, (lower, upper) in ranges.items():
        current = predictor.df[target.capitalize()].iloc[-1]
        print(f"\n{target.capitalize()}:")
        print(f"  Current: ${current:.2f}")
        print(f"  Range: ${lower:.2f} - ${upper:.2f}")

    # Multi-step predictions
    print("\n" + "=" * 80)
    print("MULTI-STEP PREDICTIONS (Next 5 Periods)")
    print("=" * 80)
    multi_step = predictor.predict_multi_step(steps=5)
    print(multi_step.to_string(index=False))

    # Specific date prediction
    print("\n" + "=" * 80)
    print("PREDICTION FOR SPECIFIC DATE")
    print("=" * 80)
    date_pred = predictor.predict_at_date('2025-10-31')
    print("\nPredicted prices for 2025-10-31:")
    for target, price in date_pred.items():
        print(f"  {target.capitalize()}: ${price:.2f}")

