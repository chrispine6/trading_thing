"""
Pattern Recognition Module
- Similar historical pattern matching
- Candlestick pattern completion probability
- Chart pattern breakout/breakdown predictions
- Support/resistance level strength
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PatternRecognizer:
    """Comprehensive pattern recognition for technical analysis"""

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

    def find_similar_patterns(self, window: int = 20, top_n: int = 5) -> List[Dict]:
        """
        Find historical patterns similar to current price action

        Args:
            window: Size of pattern window
            top_n: Number of top similar patterns to return

        Returns:
            List of similar patterns with metadata
        """
        close = self.df['Close']

        # Normalize current pattern
        current_pattern = close.iloc[-window:].values
        current_normalized = (current_pattern - current_pattern.mean()) / current_pattern.std()

        # Search for similar patterns in history
        similarities = []

        for i in range(window, len(close) - window - 20):  # Leave buffer for future outcome
            historical_pattern = close.iloc[i-window:i].values
            historical_normalized = (historical_pattern - historical_pattern.mean()) / historical_pattern.std()

            # Calculate similarity (inverse of distance)
            distance = euclidean(current_normalized, historical_normalized)
            similarity = 1 / (1 + distance)

            # Get what happened after this pattern
            future_returns = []
            for j in [1, 5, 10, 20]:
                if i + j < len(close):
                    future_return = (close.iloc[i + j] - close.iloc[i]) / close.iloc[i]
                    future_returns.append(future_return)

            similarities.append({
                'date': self.df.index[i],
                'similarity_score': float(similarity),
                'distance': float(distance),
                'future_1d_%': float(future_returns[0] * 100) if len(future_returns) > 0 else None,
                'future_5d_%': float(future_returns[1] * 100) if len(future_returns) > 1 else None,
                'future_10d_%': float(future_returns[2] * 100) if len(future_returns) > 2 else None,
                'future_20d_%': float(future_returns[3] * 100) if len(future_returns) > 3 else None
            })

        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_patterns = similarities[:top_n]

        # Calculate average expected returns
        avg_returns = {
            '1d': np.mean([p['future_1d_%'] for p in top_patterns if p['future_1d_%'] is not None]),
            '5d': np.mean([p['future_5d_%'] for p in top_patterns if p['future_5d_%'] is not None]),
            '10d': np.mean([p['future_10d_%'] for p in top_patterns if p['future_10d_%'] is not None]),
            '20d': np.mean([p['future_20d_%'] for p in top_patterns if p['future_20d_%'] is not None])
        }

        return {
            'patterns': top_patterns,
            'average_expected_returns': avg_returns,
            'bullish_ratio': sum(1 for p in top_patterns if p.get('future_5d_%', 0) > 0) / len(top_patterns)
        }

    def detect_candlestick_patterns(self) -> Dict[str, any]:
        """
        Detect common candlestick patterns and their completion probability

        Returns:
            Dictionary with detected patterns
        """
        df = self.df.iloc[-100:].copy()  # Last 100 periods

        # Calculate candle properties
        df['Body'] = df['Close'] - df['Open']
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Range'] = df['High'] - df['Low']
        df['Body_Pct'] = abs(df['Body']) / df['Range']

        patterns_detected = []

        # Get last few candles
        if len(df) >= 3:
            c0 = df.iloc[-1]  # Current
            c1 = df.iloc[-2]  # Previous
            c2 = df.iloc[-3]  # Two periods ago

            # Doji
            if c0['Body_Pct'] < 0.1:
                patterns_detected.append({
                    'pattern': 'Doji',
                    'type': 'Reversal',
                    'signal': 'Neutral/Reversal',
                    'reliability': 60
                })

            # Hammer / Hanging Man
            if (c0['Lower_Shadow'] > 2 * abs(c0['Body']) and
                c0['Upper_Shadow'] < abs(c0['Body']) * 0.5 and
                c0['Body_Pct'] < 0.4):
                if c0['Body'] > 0:
                    patterns_detected.append({
                        'pattern': 'Hammer',
                        'type': 'Bullish Reversal',
                        'signal': 'Buy',
                        'reliability': 70
                    })
                else:
                    patterns_detected.append({
                        'pattern': 'Hanging Man',
                        'type': 'Bearish Reversal',
                        'signal': 'Sell',
                        'reliability': 65
                    })

            # Shooting Star / Inverted Hammer
            if (c0['Upper_Shadow'] > 2 * abs(c0['Body']) and
                c0['Lower_Shadow'] < abs(c0['Body']) * 0.5 and
                c0['Body_Pct'] < 0.4):
                if c0['Body'] < 0:
                    patterns_detected.append({
                        'pattern': 'Shooting Star',
                        'type': 'Bearish Reversal',
                        'signal': 'Sell',
                        'reliability': 70
                    })
                else:
                    patterns_detected.append({
                        'pattern': 'Inverted Hammer',
                        'type': 'Bullish Reversal',
                        'signal': 'Buy',
                        'reliability': 60
                    })

            # Engulfing patterns
            if abs(c0['Body']) > abs(c1['Body']) * 1.5:
                if c0['Body'] > 0 and c1['Body'] < 0:
                    if c0['Open'] < c1['Close'] and c0['Close'] > c1['Open']:
                        patterns_detected.append({
                            'pattern': 'Bullish Engulfing',
                            'type': 'Bullish Reversal',
                            'signal': 'Buy',
                            'reliability': 75
                        })
                elif c0['Body'] < 0 and c1['Body'] > 0:
                    if c0['Open'] > c1['Close'] and c0['Close'] < c1['Open']:
                        patterns_detected.append({
                            'pattern': 'Bearish Engulfing',
                            'type': 'Bearish Reversal',
                            'signal': 'Sell',
                            'reliability': 75
                        })

            # Morning Star / Evening Star
            if len(df) >= 3:
                # Morning Star (bullish)
                if (c2['Body'] < 0 and abs(c2['Body']) > df['Range'].mean() * 0.5 and
                    abs(c1['Body']) < df['Range'].mean() * 0.3 and
                    c0['Body'] > 0 and abs(c0['Body']) > df['Range'].mean() * 0.5):
                    patterns_detected.append({
                        'pattern': 'Morning Star',
                        'type': 'Bullish Reversal',
                        'signal': 'Buy',
                        'reliability': 80
                    })

                # Evening Star (bearish)
                if (c2['Body'] > 0 and abs(c2['Body']) > df['Range'].mean() * 0.5 and
                    abs(c1['Body']) < df['Range'].mean() * 0.3 and
                    c0['Body'] < 0 and abs(c0['Body']) > df['Range'].mean() * 0.5):
                    patterns_detected.append({
                        'pattern': 'Evening Star',
                        'type': 'Bearish Reversal',
                        'signal': 'Sell',
                        'reliability': 80
                    })

        # Calculate overall signal
        if patterns_detected:
            bullish_count = sum(1 for p in patterns_detected if 'Bullish' in p['type'])
            bearish_count = sum(1 for p in patterns_detected if 'Bearish' in p['type'])
            avg_reliability = np.mean([p['reliability'] for p in patterns_detected])

            if bullish_count > bearish_count:
                overall_signal = 'Bullish'
            elif bearish_count > bullish_count:
                overall_signal = 'Bearish'
            else:
                overall_signal = 'Neutral'
        else:
            overall_signal = 'None'
            avg_reliability = 0

        return {
            'patterns_found': patterns_detected,
            'count': len(patterns_detected),
            'overall_signal': overall_signal,
            'average_reliability': float(avg_reliability)
        }

    def detect_chart_patterns(self) -> Dict[str, any]:
        """
        Detect chart patterns (triangles, head and shoulders, etc.)

        Returns:
            Dictionary with chart pattern analysis
        """
        close = self.df['Close'].iloc[-100:]
        high = self.df['High'].iloc[-100:]
        low = self.df['Low'].iloc[-100:]

        # Smooth the data for better peak/trough detection
        close_smooth = pd.Series(savgol_filter(close, window_length=11, polyorder=3), index=close.index)

        # Find peaks and troughs
        peaks, _ = find_peaks(close_smooth, distance=5, prominence=close.std() * 0.5)
        troughs, _ = find_peaks(-close_smooth, distance=5, prominence=close.std() * 0.5)

        patterns_detected = []

        # Triangle patterns
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Ascending Triangle: flat top, rising bottoms
            recent_peaks = peaks[-2:]
            recent_troughs = troughs[-2:]

            if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                peak_slope = close_smooth.iloc[recent_peaks[-1]] - close_smooth.iloc[recent_peaks[-2]]
                trough_slope = close_smooth.iloc[recent_troughs[-1]] - close_smooth.iloc[recent_troughs[-2]]

                if abs(peak_slope) < close.std() * 0.3 and trough_slope > 0:
                    patterns_detected.append({
                        'pattern': 'Ascending Triangle',
                        'type': 'Bullish Continuation',
                        'breakout_target': float(close.iloc[-1] * 1.05),
                        'probability': 70
                    })

                # Descending Triangle: flat bottom, declining tops
                elif abs(trough_slope) < close.std() * 0.3 and peak_slope < 0:
                    patterns_detected.append({
                        'pattern': 'Descending Triangle',
                        'type': 'Bearish Continuation',
                        'breakout_target': float(close.iloc[-1] * 0.95),
                        'probability': 70
                    })

                # Symmetrical Triangle: converging lines
                elif peak_slope < 0 and trough_slope > 0:
                    patterns_detected.append({
                        'pattern': 'Symmetrical Triangle',
                        'type': 'Neutral (Continuation)',
                        'breakout_target': None,
                        'probability': 60
                    })

        # Head and Shoulders
        if len(peaks) >= 3:
            recent_peaks = peaks[-3:]
            peak_values = close_smooth.iloc[recent_peaks]

            # Head and Shoulders: middle peak is highest
            if (peak_values.iloc[1] > peak_values.iloc[0] and
                peak_values.iloc[1] > peak_values.iloc[2] and
                abs(peak_values.iloc[0] - peak_values.iloc[2]) < close.std() * 0.5):

                neckline = close_smooth.iloc[troughs[-2:]].mean() if len(troughs) >= 2 else close.mean()
                target = float(close.iloc[-1] - (peak_values.iloc[1] - neckline))

                patterns_detected.append({
                    'pattern': 'Head and Shoulders',
                    'type': 'Bearish Reversal',
                    'breakout_target': target,
                    'probability': 75
                })

            # Inverse Head and Shoulders
            elif (peak_values.iloc[1] < peak_values.iloc[0] and
                  peak_values.iloc[1] < peak_values.iloc[2] and
                  abs(peak_values.iloc[0] - peak_values.iloc[2]) < close.std() * 0.5):

                neckline = close_smooth.iloc[peaks[-2:]].mean() if len(peaks) >= 2 else close.mean()
                target = float(close.iloc[-1] + (neckline - peak_values.iloc[1]))

                patterns_detected.append({
                    'pattern': 'Inverse Head and Shoulders',
                    'type': 'Bullish Reversal',
                    'breakout_target': target,
                    'probability': 75
                })

        # Double Top/Bottom
        if len(peaks) >= 2:
            last_two_peaks = close_smooth.iloc[peaks[-2:]]
            if abs(last_two_peaks.iloc[0] - last_two_peaks.iloc[1]) < close.std() * 0.3:
                patterns_detected.append({
                    'pattern': 'Double Top',
                    'type': 'Bearish Reversal',
                    'breakout_target': float(close.iloc[-1] * 0.95),
                    'probability': 70
                })

        if len(troughs) >= 2:
            last_two_troughs = close_smooth.iloc[troughs[-2:]]
            if abs(last_two_troughs.iloc[0] - last_two_troughs.iloc[1]) < close.std() * 0.3:
                patterns_detected.append({
                    'pattern': 'Double Bottom',
                    'type': 'Bullish Reversal',
                    'breakout_target': float(close.iloc[-1] * 1.05),
                    'probability': 70
                })

        return {
            'patterns_found': patterns_detected,
            'count': len(patterns_detected),
            'peaks_identified': len(peaks),
            'troughs_identified': len(troughs)
        }

    def identify_support_resistance(self, window: int = 100) -> Dict[str, any]:
        """
        Identify support and resistance levels with strength metrics

        Args:
            window: Lookback window for analysis

        Returns:
            Dictionary with support/resistance levels
        """
        df = self.df.iloc[-window:]
        close = df['Close']
        high = df['High']
        low = df['Low']

        # Find all local maxima and minima
        peaks, peak_props = find_peaks(high.values, distance=5, prominence=close.std() * 0.3)
        troughs, trough_props = find_peaks(-low.values, distance=5, prominence=close.std() * 0.3)

        # Cluster levels (group nearby levels)
        def cluster_levels(levels, tolerance=0.02):
            if len(levels) == 0:
                return []

            levels_sorted = sorted(levels)
            clusters = []
            current_cluster = [levels_sorted[0]]

            for level in levels_sorted[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]

            clusters.append(current_cluster)
            return clusters

        # Get resistance levels (from peaks)
        resistance_levels = high.iloc[peaks].values
        resistance_clusters = cluster_levels(resistance_levels)

        # Get support levels (from troughs)
        support_levels = low.iloc[troughs].values
        support_clusters = cluster_levels(support_levels)

        # Calculate strength of each level
        def calculate_strength(cluster, price_series):
            avg_level = np.mean(cluster)
            touches = sum(1 for price in price_series if abs(price - avg_level) / avg_level < 0.01)
            recency = 1.0  # Could weight by recency
            strength = touches * recency * 10
            return min(strength, 100)

        # Format resistance levels
        resistances = []
        current_price = close.iloc[-1]

        for cluster in resistance_clusters:
            avg_level = np.mean(cluster)
            if avg_level > current_price:  # Only levels above current price
                strength = calculate_strength(cluster, high)
                distance_pct = (avg_level - current_price) / current_price * 100

                resistances.append({
                    'level': float(avg_level),
                    'strength': float(strength),
                    'touches': len(cluster),
                    'distance_%': float(distance_pct)
                })

        resistances.sort(key=lambda x: x['distance_%'])

        # Format support levels
        supports = []
        for cluster in support_clusters:
            avg_level = np.mean(cluster)
            if avg_level < current_price:  # Only levels below current price
                strength = calculate_strength(cluster, low)
                distance_pct = (current_price - avg_level) / current_price * 100

                supports.append({
                    'level': float(avg_level),
                    'strength': float(strength),
                    'touches': len(cluster),
                    'distance_%': float(distance_pct)
                })

        supports.sort(key=lambda x: x['distance_%'])

        # Calculate current position
        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None

        if nearest_support and nearest_resistance:
            range_position = (current_price - nearest_support['level']) / (nearest_resistance['level'] - nearest_support['level'])
        else:
            range_position = 0.5

        return {
            'current_price': float(current_price),
            'support_levels': supports[:5],  # Top 5 nearest
            'resistance_levels': resistances[:5],  # Top 5 nearest
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'position_in_range_%': float(range_position * 100)
        }

    def get_comprehensive_pattern_report(self) -> Dict[str, any]:
        """Generate comprehensive pattern recognition report"""
        return {
            'similar_patterns': self.find_similar_patterns(),
            'candlestick_patterns': self.detect_candlestick_patterns(),
            'chart_patterns': self.detect_chart_patterns(),
            'support_resistance': self.identify_support_resistance()
        }


if __name__ == "__main__":
    # Example usage
    recognizer = PatternRecognizer('../../data/aapl.us.csv')

    print("=" * 80)
    print("SIMILAR HISTORICAL PATTERNS")
    print("=" * 80)
    similar = recognizer.find_similar_patterns(window=20, top_n=5)
    print(f"\nBullish Ratio: {similar['bullish_ratio']:.1%}")
    print(f"\nAverage Expected Returns:")
    for period, ret in similar['average_expected_returns'].items():
        print(f"  {period}: {ret:+.2f}%")
    print(f"\nTop 5 Similar Patterns:")
    for i, pattern in enumerate(similar['patterns'], 1):
        print(f"\n  {i}. Date: {pattern['date'].strftime('%Y-%m-%d')}")
        print(f"     Similarity: {pattern['similarity_score']:.4f}")
        print(f"     Future 5d: {pattern['future_5d_%']:+.2f}%")

    print("\n" + "=" * 80)
    print("CANDLESTICK PATTERNS")
    print("=" * 80)
    candles = recognizer.detect_candlestick_patterns()
    print(f"\nPatterns Found: {candles['count']}")
    print(f"Overall Signal: {candles['overall_signal']}")
    print(f"Average Reliability: {candles['average_reliability']:.1f}%")
    print("\nDetected Patterns:")
    for pattern in candles['patterns_found']:
        print(f"\n  • {pattern['pattern']}")
        print(f"    Type: {pattern['type']}")
        print(f"    Signal: {pattern['signal']}")
        print(f"    Reliability: {pattern['reliability']}%")

    print("\n" + "=" * 80)
    print("CHART PATTERNS")
    print("=" * 80)
    charts = recognizer.detect_chart_patterns()
    print(f"\nPatterns Found: {charts['count']}")
    print(f"Peaks Identified: {charts['peaks_identified']}")
    print(f"Troughs Identified: {charts['troughs_identified']}")
    print("\nDetected Patterns:")
    for pattern in charts['patterns_found']:
        print(f"\n  • {pattern['pattern']}")
        print(f"    Type: {pattern['type']}")
        if pattern['breakout_target']:
            print(f"    Target: ${pattern['breakout_target']:.2f}")
        print(f"    Probability: {pattern['probability']}%")

    print("\n" + "=" * 80)
    print("SUPPORT & RESISTANCE LEVELS")
    print("=" * 80)
    sr = recognizer.identify_support_resistance()
    print(f"\nCurrent Price: ${sr['current_price']:.2f}")
    print(f"Position in Range: {sr['position_in_range_%']:.1f}%")

    print("\nKey Resistance Levels:")
    for i, level in enumerate(sr['resistance_levels'][:3], 1):
        print(f"  {i}. ${level['level']:.2f} (+{level['distance_%']:.2f}%) - Strength: {level['strength']:.0f}, Touches: {level['touches']}")

    print("\nKey Support Levels:")
    for i, level in enumerate(sr['support_levels'][:3], 1):
        print(f"  {i}. ${level['level']:.2f} (-{level['distance_%']:.2f}%) - Strength: {level['strength']:.0f}, Touches: {level['touches']}")
"""
Volatility Predictions Module
- Future volatility estimates (GARCH models)
- Volatility clustering patterns
- Breakout probability from consolidation ranges
- Risk metrics (Value at Risk, Expected Shortfall)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import ARCH for GARCH models
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch package not available. Install with: pip install arch")


class VolatilityPredictor:
    """Comprehensive volatility prediction and risk analysis"""

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

        # Calculate returns
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))

    def estimate_historical_volatility(self, window: int = 20) -> Dict[str, float]:
        """
        Calculate historical volatility using multiple methods

        Args:
            window: Lookback window for volatility calculation

        Returns:
            Dictionary with volatility estimates
        """
        returns = self.df['Returns'].dropna()
        log_returns = self.df['Log_Returns'].dropna()

        # Standard deviation of returns (annualized)
        daily_vol = returns.iloc[-window:].std()
        annual_vol = daily_vol * np.sqrt(252)  # 252 trading days

        # Parkinson volatility (uses High-Low)
        high = self.df['High'].iloc[-window:]
        low = self.df['Low'].iloc[-window:]
        parkinson_vol = np.sqrt(np.sum(np.log(high / low) ** 2) / (4 * window * np.log(2)))
        parkinson_annual = parkinson_vol * np.sqrt(252)

        # Garman-Klass volatility (uses OHLC)
        open_price = self.df['Open'].iloc[-window:]
        close = self.df['Close'].iloc[-window:]

        gk_vol = np.sqrt(
            np.mean(0.5 * np.log(high / low) ** 2 -
                   (2 * np.log(2) - 1) * np.log(close / open_price) ** 2)
        )
        gk_annual = gk_vol * np.sqrt(252)

        # Rolling volatility trend
        rolling_vol = returns.rolling(window=window).std()
        vol_trend = (rolling_vol.iloc[-1] - rolling_vol.iloc[-window]) / rolling_vol.iloc[-window] if len(rolling_vol) >= window else 0

        return {
            'daily_volatility_%': float(daily_vol * 100),
            'annual_volatility_%': float(annual_vol * 100),
            'parkinson_annual_%': float(parkinson_annual * 100),
            'garman_klass_annual_%': float(gk_annual * 100),
            'volatility_trend_%': float(vol_trend * 100),
            'volatility_state': self._classify_volatility(annual_vol)
        }

    def _classify_volatility(self, annual_vol: float) -> str:
        """Classify volatility level"""
        vol_pct = annual_vol * 100
        if vol_pct < 15:
            return "Very Low"
        elif vol_pct < 25:
            return "Low"
        elif vol_pct < 35:
            return "Moderate"
        elif vol_pct < 50:
            return "High"
        else:
            return "Very High"

    def predict_garch_volatility(self, horizon: int = 5) -> Dict[str, any]:
        """
        Predict future volatility using GARCH model

        Args:
            horizon: Number of periods to forecast

        Returns:
            Dictionary with GARCH predictions
        """
        if not ARCH_AVAILABLE:
            return self._fallback_volatility_forecast(horizon)

        try:
            returns = self.df['Returns'].dropna().iloc[-500:] * 100  # Convert to percentage

            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted = model.fit(disp='off', show_warning=False)

            # Forecast volatility
            forecast = fitted.forecast(horizon=horizon)
            predicted_vol = np.sqrt(forecast.variance.values[-1, :])

            # Annualize
            predicted_annual = predicted_vol * np.sqrt(252) / 100

            forecasts = []
            for i, vol in enumerate(predicted_annual):
                forecasts.append({
                    'period': i + 1,
                    'predicted_volatility_%': float(vol * 100)
                })

            return {
                'model': 'GARCH(1,1)',
                'forecasts': forecasts,
                'mean_forecast_%': float(np.mean(predicted_annual) * 100),
                'current_volatility_%': float(self.estimate_historical_volatility()['annual_volatility_%'])
            }
        except Exception as e:
            print(f"GARCH model failed: {e}. Using fallback method.")
            return self._fallback_volatility_forecast(horizon)

    def _fallback_volatility_forecast(self, horizon: int) -> Dict[str, any]:
        """Fallback method using EWMA"""
        returns = self.df['Returns'].dropna()

        # EWMA volatility
        ewma_vol = returns.ewm(span=20).std().iloc[-1]
        annual_vol = ewma_vol * np.sqrt(252)

        # Simple persistence forecast
        forecasts = []
        for i in range(horizon):
            # Assume volatility mean-reverts slowly
            long_term_vol = returns.iloc[-252:].std() * np.sqrt(252) if len(returns) >= 252 else annual_vol
            weight = 0.95 ** (i + 1)
            forecast_vol = weight * annual_vol + (1 - weight) * long_term_vol

            forecasts.append({
                'period': i + 1,
                'predicted_volatility_%': float(forecast_vol * 100)
            })

        return {
            'model': 'EWMA',
            'forecasts': forecasts,
            'mean_forecast_%': float(np.mean([f['predicted_volatility_%'] for f in forecasts])),
            'current_volatility_%': float(annual_vol * 100)
        }

    def detect_volatility_clustering(self, threshold: float = 1.5) -> Dict[str, any]:
        """
        Detect volatility clustering patterns

        Args:
            threshold: Threshold for high volatility (multiple of median)

        Returns:
            Dictionary with clustering information
        """
        returns = self.df['Returns'].dropna()

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()
        median_vol = rolling_vol.median()

        # Identify high volatility periods
        high_vol_mask = rolling_vol > (median_vol * threshold)

        # Find clusters
        clusters = []
        in_cluster = False
        cluster_start = None

        for date, is_high_vol in high_vol_mask.items():
            if is_high_vol and not in_cluster:
                in_cluster = True
                cluster_start = date
            elif not is_high_vol and in_cluster:
                in_cluster = False
                clusters.append((cluster_start, date))

        # Current volatility regime
        current_vol = rolling_vol.iloc[-1]
        vol_ratio = current_vol / median_vol

        # Autocorrelation of squared returns (volatility clustering test)
        squared_returns = returns ** 2
        autocorr_lag1 = squared_returns.autocorr(lag=1)
        autocorr_lag5 = squared_returns.autocorr(lag=5)

        return {
            'clustering_detected': bool(autocorr_lag1 > 0.1),
            'autocorr_lag1': float(autocorr_lag1),
            'autocorr_lag5': float(autocorr_lag5),
            'current_volatility_ratio': float(vol_ratio),
            'volatility_regime': 'High' if vol_ratio > threshold else 'Normal' if vol_ratio > 0.7 else 'Low',
            'num_clusters': len(clusters),
            'recent_clusters': len([c for c in clusters if (self.df.index[-1] - c[1]).days < 90])
        }

    def calculate_breakout_probability(self, consolidation_period: int = 20) -> Dict[str, any]:
        """
        Calculate probability of breakout from consolidation range

        Args:
            consolidation_period: Period to analyze for consolidation

        Returns:
            Dictionary with breakout analysis
        """
        close = self.df['Close'].iloc[-consolidation_period:]
        high = self.df['High'].iloc[-consolidation_period:]
        low = self.df['Low'].iloc[-consolidation_period:]
        volume = self.df['Volume'].iloc[-consolidation_period:]

        # Calculate range
        price_range = high.max() - low.min()
        current_price = close.iloc[-1]
        range_position = (current_price - low.min()) / price_range

        # Volatility compression
        recent_vol = close.pct_change().std()
        historical_vol = self.df['Close'].iloc[-100:].pct_change().std() if len(self.df) >= 100 else recent_vol
        vol_ratio = recent_vol / historical_vol

        # Bollinger Bands squeeze
        sma = close.rolling(window=20).mean().iloc[-1]
        std = close.rolling(window=20).std().iloc[-1]
        bb_width = (std / sma) * 100

        # Volume analysis
        avg_volume = volume.mean()
        recent_volume = volume.iloc[-5:].mean()
        volume_ratio = recent_volume / avg_volume

        # Calculate breakout probability
        probability_score = 0
        factors = []

        # Volatility compression suggests imminent breakout
        if vol_ratio < 0.7:
            probability_score += 30
            factors.append("Volatility compression detected")

        # Narrow Bollinger Bands
        if bb_width < 2:
            probability_score += 25
            factors.append("Narrow Bollinger Bands")

        # Volume increase
        if volume_ratio > 1.2:
            probability_score += 20
            factors.append("Volume increasing")

        # Price near range boundary
        if range_position > 0.8 or range_position < 0.2:
            probability_score += 15
            factors.append(f"Price near {'upper' if range_position > 0.8 else 'lower'} boundary")

        # Time factor (longer consolidation = higher breakout probability)
        if consolidation_period >= 30:
            probability_score += 10
            factors.append("Extended consolidation period")

        return {
            'breakout_probability_%': min(probability_score, 100),
            'consolidation_range': (float(low.min()), float(high.max())),
            'range_width_%': float((price_range / current_price) * 100),
            'price_position_in_range_%': float(range_position * 100),
            'volatility_ratio': float(vol_ratio),
            'volume_ratio': float(volume_ratio),
            'bollinger_width_%': float(bb_width),
            'factors': factors,
            'likely_direction': 'Upward' if range_position > 0.6 else 'Downward' if range_position < 0.4 else 'Uncertain'
        }

    def calculate_var_es(self, confidence: float = 0.95, horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk and Expected Shortfall

        Args:
            confidence: Confidence level (default 0.95 for 95%)
            horizon: Time horizon in days

        Returns:
            Dictionary with VaR and ES
        """
        returns = self.df['Returns'].dropna()
        current_price = self.df['Close'].iloc[-1]

        # Historical VaR (parametric)
        mean_return = returns.mean()
        std_return = returns.std()
        var_parametric = stats.norm.ppf(1 - confidence, mean_return, std_return)

        # Historical VaR (empirical)
        var_historical = returns.quantile(1 - confidence)

        # Expected Shortfall (CVaR) - average of losses beyond VaR
        es_historical = returns[returns <= var_historical].mean()

        # Scale to horizon and dollar amounts
        horizon_factor = np.sqrt(horizon)

        var_parametric_scaled = var_parametric * horizon_factor
        var_historical_scaled = var_historical * horizon_factor
        es_scaled = es_historical * horizon_factor

        # Convert to dollar values
        var_parametric_dollar = current_price * var_parametric_scaled
        var_historical_dollar = current_price * var_historical_scaled
        es_dollar = current_price * es_scaled

        return {
            'confidence_level': confidence,
            'horizon_days': horizon,
            'current_price': float(current_price),
            'var_parametric_%': float(var_parametric_scaled * 100),
            'var_parametric_$': float(var_parametric_dollar),
            'var_historical_%': float(var_historical_scaled * 100),
            'var_historical_$': float(var_historical_dollar),
            'expected_shortfall_%': float(es_scaled * 100),
            'expected_shortfall_$': float(es_dollar),
            'interpretation': f"95% confident that loss will not exceed ${abs(var_historical_dollar):.2f} ({abs(var_historical_scaled)*100:.2f}%) over {horizon} day(s)"
        }

    def get_comprehensive_volatility_report(self) -> Dict[str, any]:
        """Generate comprehensive volatility analysis report"""
        return {
            'historical_volatility': self.estimate_historical_volatility(),
            'garch_forecast': self.predict_garch_volatility(horizon=5),
            'clustering': self.detect_volatility_clustering(),
            'breakout_analysis': self.calculate_breakout_probability(),
            'risk_metrics': {
                '1day_95%': self.calculate_var_es(confidence=0.95, horizon=1),
                '5day_95%': self.calculate_var_es(confidence=0.95, horizon=5),
                '1day_99%': self.calculate_var_es(confidence=0.99, horizon=1)
            }
        }


if __name__ == "__main__":
    # Example usage
    predictor = VolatilityPredictor('../../data/aapl.us.csv')

    print("=" * 80)
    print("HISTORICAL VOLATILITY")
    print("=" * 80)
    hist_vol = predictor.estimate_historical_volatility()
    print(f"\nDaily Volatility: {hist_vol['daily_volatility_%']:.2f}%")
    print(f"Annual Volatility: {hist_vol['annual_volatility_%']:.2f}%")
    print(f"Parkinson (High-Low): {hist_vol['parkinson_annual_%']:.2f}%")
    print(f"Garman-Klass (OHLC): {hist_vol['garman_klass_annual_%']:.2f}%")
    print(f"Volatility Trend: {hist_vol['volatility_trend_%']:+.2f}%")
    print(f"State: {hist_vol['volatility_state']}")

    print("\n" + "=" * 80)
    print("GARCH VOLATILITY FORECAST")
    print("=" * 80)
    garch = predictor.predict_garch_volatility(horizon=5)
    print(f"\nModel: {garch['model']}")
    print(f"Current Volatility: {garch['current_volatility_%']:.2f}%")
    print(f"Mean Forecast: {garch['mean_forecast_%']:.2f}%")
    print("\nPeriod-by-Period Forecast:")
    for forecast in garch['forecasts']:
        print(f"  Period {forecast['period']}: {forecast['predicted_volatility_%']:.2f}%")

    print("\n" + "=" * 80)
    print("VOLATILITY CLUSTERING")
    print("=" * 80)
    clustering = predictor.detect_volatility_clustering()
    print(f"\nClustering Detected: {clustering['clustering_detected']}")
    print(f"Autocorrelation (Lag 1): {clustering['autocorr_lag1']:.4f}")
    print(f"Autocorrelation (Lag 5): {clustering['autocorr_lag5']:.4f}")
    print(f"Current Volatility Ratio: {clustering['current_volatility_ratio']:.2f}x")
    print(f"Volatility Regime: {clustering['volatility_regime']}")
    print(f"Total Clusters Found: {clustering['num_clusters']}")
    print(f"Recent Clusters (90 days): {clustering['recent_clusters']}")

    print("\n" + "=" * 80)
    print("BREAKOUT PROBABILITY")
    print("=" * 80)
    breakout = predictor.calculate_breakout_probability()
    print(f"\nBreakout Probability: {breakout['breakout_probability_%']:.1f}%")
    print(f"Consolidation Range: ${breakout['consolidation_range'][0]:.2f} - ${breakout['consolidation_range'][1]:.2f}")
    print(f"Range Width: {breakout['range_width_%']:.2f}%")
    print(f"Price Position: {breakout['price_position_in_range_%']:.1f}% of range")
    print(f"Volatility Ratio: {breakout['volatility_ratio']:.2f}")
    print(f"Volume Ratio: {breakout['volume_ratio']:.2f}")
    print(f"Likely Direction: {breakout['likely_direction']}")
    print("\nFactors:")
    for factor in breakout['factors']:
        print(f"  • {factor}")

    print("\n" + "=" * 80)
    print("RISK METRICS (VaR & Expected Shortfall)")
    print("=" * 80)

    # 1-day 95% VaR
    var_1d = predictor.calculate_var_es(confidence=0.95, horizon=1)
    print(f"\n1-Day VaR (95% confidence):")
    print(f"  Parametric: {var_1d['var_parametric_%']:.2f}% (${abs(var_1d['var_parametric_$']):.2f})")
    print(f"  Historical: {var_1d['var_historical_%']:.2f}% (${abs(var_1d['var_historical_$']):.2f})")
    print(f"  Expected Shortfall: {var_1d['expected_shortfall_%']:.2f}% (${abs(var_1d['expected_shortfall_$']):.2f})")

    # 5-day 95% VaR
    var_5d = predictor.calculate_var_es(confidence=0.95, horizon=5)
    print(f"\n5-Day VaR (95% confidence):")
    print(f"  Parametric: {var_5d['var_parametric_%']:.2f}% (${abs(var_5d['var_parametric_$']):.2f})")
    print(f"  Historical: {var_5d['var_historical_%']:.2f}% (${abs(var_5d['var_historical_$']):.2f})")
    print(f"  Expected Shortfall: {var_5d['expected_shortfall_%']:.2f}% (${abs(var_5d['expected_shortfall_$']):.2f})")

    # 1-day 99% VaR
    var_99 = predictor.calculate_var_es(confidence=0.99, horizon=1)
    print(f"\n1-Day VaR (99% confidence):")
    print(f"  Historical: {var_99['var_historical_%']:.2f}% (${abs(var_99['var_historical_$']):.2f})")
