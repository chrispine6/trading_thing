import pandas as pd
import numpy as np


class CandlestickAnalyzer:
    """
    A comprehensive toolkit for analyzing candlestick patterns and trends.
    Expects a DataFrame with columns: ['open', 'high', 'low', 'close']
    """

    def __init__(self, df):
        """
        Initialize with candlestick data
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        """
        self.df = df.copy()
        # Add convenience column for candle color
        self.df['is_green'] = self.df['close'] > self.df['open']

    # ==================== 1. CANDLE BODY STRENGTH ====================

    def body_size(self):
        """Calculate absolute body size for each candle"""
        return abs(self.df['close'] - self.df['open'])

    def body_percentage(self):
        """Calculate body size as percentage of opening price"""
        body = self.body_size()
        return (body / self.df['open']) * 100

    def average_body_percentage(self, window=10):
        """Rolling average body percentage to track conviction trends"""
        return self.body_percentage().rolling(window=window).mean()

    # ==================== 2. WIN/LOSS RATIO ====================

    def bull_bear_ratio(self, window=20):
        """
        Calculate percentage of bullish candles in rolling window
        Returns: bull_ratio, bear_ratio as Series
        """
        bull_ratio = self.df['is_green'].rolling(window=window).sum() / window * 100
        bear_ratio = 100 - bull_ratio
        return bull_ratio, bear_ratio

    def directional_bias(self, window=20):
        """
        Simple interpretation of bull/bear ratio
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        bull_ratio, _ = self.bull_bear_ratio(window)
        latest = bull_ratio.iloc[-1]

        if latest > 65:
            return 'BULLISH'
        elif latest < 35:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    # ==================== 3. HIGHER HIGHS & HIGHER LOWS ====================

    def identify_swings(self, window=5):
        """
        Identify swing highs and lows
        A swing high: highest point in window
        A swing low: lowest point in window
        """
        self.df['swing_high'] = self.df['high'].rolling(window=window, center=True).max() == self.df['high']
        self.df['swing_low'] = self.df['low'].rolling(window=window, center=True).min() == self.df['low']
        return self.df[['swing_high', 'swing_low']]

    def higher_highs_lows(self, lookback=3):
        """
        Check for higher highs and higher lows pattern
        Returns: trend direction and strength score
        """
        # Get recent highs and lows
        recent_highs = self.df['high'].tail(lookback).values
        recent_lows = self.df['low'].tail(lookback).values

        # Count consecutive higher highs
        hh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i] > recent_highs[i - 1])

        # Count consecutive higher lows
        hl_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i] > recent_lows[i - 1])

        # Count consecutive lower highs
        lh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i] < recent_highs[i - 1])

        # Count consecutive lower lows
        ll_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i] < recent_lows[i - 1])

        # Determine trend
        uptrend_score = hh_count + hl_count
        downtrend_score = lh_count + ll_count

        if uptrend_score > downtrend_score:
            return 'UPTREND', uptrend_score
        elif downtrend_score > uptrend_score:
            return 'DOWNTREND', downtrend_score
        else:
            return 'SIDEWAYS', 0

    # ==================== 4. WICK ANALYSIS ====================

    def wick_sizes(self):
        """Calculate upper and lower wick sizes"""
        upper_wick = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_wick = self.df[['open', 'close']].min(axis=1) - self.df['low']
        return upper_wick, lower_wick

    def rejection_ratio(self):
        """
        Calculate rejection at top (upper wick / total range)
        High ratio = strong rejection at top (bearish)
        """
        upper_wick, _ = self.wick_sizes()
        total_range = self.df['high'] - self.df['low']
        # Avoid division by zero
        return np.where(total_range > 0, upper_wick / total_range, 0)

    def wick_analysis(self, index=-1):
        """
        Analyze wicks for specific candle (default: most recent)
        """
        upper_wick, lower_wick = self.wick_sizes()
        body = self.body_size()

        uw = upper_wick.iloc[index]
        lw = lower_wick.iloc[index]
        b = body.iloc[index]

        analysis = {
            'upper_wick': uw,
            'lower_wick': lw,
            'body': b,
            'upper_dominance': uw > b and uw > lw,
            'lower_dominance': lw > b and lw > uw,
            'interpretation': ''
        }

        if analysis['upper_dominance']:
            analysis['interpretation'] = 'Strong rejection at top - bearish signal'
        elif analysis['lower_dominance']:
            analysis['interpretation'] = 'Strong rejection at bottom - bullish signal'
        else:
            analysis['interpretation'] = 'Balanced or body-dominated candle'

        return analysis

    # ==================== 5. MOMENTUM ====================

    def momentum(self, periods=5):
        """
        Calculate momentum as percentage change over N periods
        """
        return ((self.df['close'] - self.df['close'].shift(periods)) /
                self.df['close'].shift(periods) * 100)

    def acceleration(self, periods=5):
        """
        Calculate if momentum is increasing or decreasing
        Compares recent momentum to previous momentum
        """
        mom = self.momentum(periods)
        return mom.diff()  # Positive = accelerating, Negative = decelerating

    # ==================== 6. PULLBACK MAGNITUDE ====================

    def pullback_percentage(self, window=20):
        """
        Calculate pullback from recent high
        """
        recent_high = self.df['high'].rolling(window=window).max()
        return ((recent_high - self.df['low']) / recent_high * 100)

    def pullback_health(self, window=20):
        """
        Assess if pullback is healthy (20-40%) or concerning (>50%)
        """
        pullback = self.pullback_percentage(window).iloc[-1]

        if pullback < 20:
            return 'MINIMAL', pullback
        elif 20 <= pullback < 40:
            return 'HEALTHY', pullback
        elif 40 <= pullback < 50:
            return 'MODERATE', pullback
        else:
            return 'DEEP', pullback

    # ==================== 7. AVERAGE TRUE RANGE (ATR) ====================

    def true_range(self):
        """Calculate True Range for each candle"""
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())

        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    def atr(self, period=14):
        """Calculate Average True Range"""
        tr = self.true_range()
        return tr.rolling(window=period).mean()

    def atr_trend(self, period=14):
        """
        Determine if volatility is increasing or decreasing
        """
        atr_values = self.atr(period)
        recent_atr = atr_values.iloc[-5:].mean()
        older_atr = atr_values.iloc[-15:-5].mean()

        if recent_atr > older_atr * 1.1:
            return 'INCREASING', recent_atr
        elif recent_atr < older_atr * 0.9:
            return 'DECREASING', recent_atr
        else:
            return 'STABLE', recent_atr

    # ==================== 8. TREND CONSISTENCY ====================

    def consistency_score(self, window=20):
        """
        Calculate trend consistency
        Ratio of movement in dominant direction vs total movement
        """
        recent = self.df.tail(window)

        # Calculate body sizes for green and red candles
        green_bodies = recent[recent['is_green']].apply(
            lambda row: abs(row['close'] - row['open']), axis=1
        ).sum()

        red_bodies = recent[~recent['is_green']].apply(
            lambda row: abs(row['close'] - row['open']), axis=1
        ).sum()

        total_bodies = green_bodies + red_bodies

        if total_bodies == 0:
            return 0

        # Return consistency in dominant direction
        dominant = max(green_bodies, red_bodies)
        return (dominant / total_bodies) * 100

    # ==================== 9. MOVING AVERAGES ====================

    def moving_averages(self, short=10, long=50):
        """Calculate short and long moving averages"""
        self.df['ma_short'] = self.df['close'].rolling(window=short).mean()
        self.df['ma_long'] = self.df['close'].rolling(window=long).mean()
        return self.df[['ma_short', 'ma_long']]

    def ma_trend(self, short=10, long=50):
        """
        Determine trend based on MA relationships
        """
        self.moving_averages(short, long)

        current_price = self.df['close'].iloc[-1]
        ma_short = self.df['ma_short'].iloc[-1]
        ma_long = self.df['ma_long'].iloc[-1]

        if pd.isna(ma_short) or pd.isna(ma_long):
            return 'INSUFFICIENT_DATA'

        if current_price > ma_short > ma_long:
            return 'STRONG_UPTREND'
        elif current_price > ma_short and ma_short < ma_long:
            return 'EARLY_UPTREND'
        elif current_price < ma_short < ma_long:
            return 'STRONG_DOWNTREND'
        elif current_price < ma_short and ma_short > ma_long:
            return 'EARLY_DOWNTREND'
        else:
            return 'MIXED'

    # ==================== 10. COMPOSITE TREND STRENGTH ====================

    def trend_strength_score(self, window=20):
        """
        Calculate composite trend strength score (0-100)
        Combines multiple factors with weighted importance
        """
        # Get bull ratio
        bull_ratio, _ = self.bull_bear_ratio(window)
        bull_score = bull_ratio.iloc[-1] if not pd.isna(bull_ratio.iloc[-1]) else 50

        # Get average body percentage
        avg_body = self.body_percentage().tail(window).mean()
        body_score = min(avg_body * 10, 100)  # Scale up, cap at 100

        # Get momentum
        mom = self.momentum(window).iloc[-1]
        momentum_score = min(abs(mom) * 2, 100)  # Scale up, cap at 100

        # Get consistency
        consistency = self.consistency_score(window)

        # Get higher highs/lows
        _, hh_ll_count = self.higher_highs_lows(lookback=min(5, window))
        hh_ll_score = min(hh_ll_count * 20, 100)  # Scale up, cap at 100

        # Weighted combination
        score = (
                0.30 * bull_score +
                0.20 * body_score +
                0.20 * momentum_score +
                0.15 * consistency +
                0.15 * hh_ll_score
        )

        return score

    def interpret_trend_strength(self, window=20):
        """
        Get trend strength with interpretation
        """
        score = self.trend_strength_score(window)

        if score > 70:
            strength = 'STRONG'
        elif score > 40:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

        # Determine direction
        bull_ratio, _ = self.bull_bear_ratio(window)
        direction = 'BULLISH' if bull_ratio.iloc[-1] > 50 else 'BEARISH'

        return {
            'score': score,
            'strength': strength,
            'direction': direction,
            'interpretation': f'{strength} {direction} TREND'
        }

    # ==================== COMPREHENSIVE ANALYSIS ====================

    def analyze(self, window=20):
        """
        Run comprehensive analysis and return all metrics
        """
        report = {
            'trend_strength': self.interpret_trend_strength(window),
            'directional_bias': self.directional_bias(window),
            'ma_trend': self.ma_trend(),
            'higher_highs_lows': self.higher_highs_lows(),
            'consistency': self.consistency_score(window),
            'momentum': self.momentum(5).iloc[-1],
            'pullback': self.pullback_health(window),
            'atr_trend': self.atr_trend(),
            'recent_wick': self.wick_analysis(),
        }

        return report

# Example usage: