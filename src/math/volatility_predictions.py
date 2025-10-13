"""
Volatility Predictions Module
- Future volatility estimates (GARCH models)
- Volatility clustering patterns
- Breakout probability from consolidation ranges
- Risk metrics (Value at Risk, Expected Shortfall)
- Comprehensive volatility measurements and estimators
"""

import pandas as pd
import numpy as np
from scipy import stats
from arch import arch_model
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


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
        self.df['Returns'] = self.df['Close'].pct_change() * 100  # Returns in percentage

        # Calculate log returns
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1)) * 100

    def predict_volatility_garch(self, horizon: int = 5) -> Dict[str, any]:
        """
        Predict future volatility using GARCH(1,1) model

        Args:
            horizon: Forecast horizon in periods

        Returns:
            Dictionary with volatility forecasts
        """
        returns = self.df['Returns'].dropna()

        try:
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            fitted_model = model.fit(disp='off')

            # Forecast volatility
            forecast = fitted_model.forecast(horizon=horizon)
            variance_forecast = forecast.variance.values[-1, :]

            # Convert to volatility (standard deviation)
            volatility_forecast = np.sqrt(variance_forecast)

            result = {
                'method': 'GARCH(1,1)',
                'current_volatility': float(returns.iloc[-20:].std()),
                'forecasts': [float(vol) for vol in volatility_forecast],
                'mean_forecast': float(np.mean(volatility_forecast)),
                'model_params': {
                    'omega': float(fitted_model.params['omega']),
                    'alpha': float(fitted_model.params['alpha[1]']),
                    'beta': float(fitted_model.params['beta[1]'])
                },
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic)
            }

        except Exception as e:
            # Fallback to simple historical volatility
            result = {
                'method': 'Historical (Fallback)',
                'current_volatility': float(returns.iloc[-20:].std()),
                'forecasts': [float(returns.iloc[-20:].std())] * horizon,
                'mean_forecast': float(returns.iloc[-20:].std()),
                'error': str(e)
            }

        return result

    def predict_volatility_ewma(self, lambda_param: float = 0.94, horizon: int = 5) -> Dict[str, any]:
        """
        Predict volatility using Exponentially Weighted Moving Average (EWMA)

        Args:
            lambda_param: Decay factor (typically 0.94 for daily data)
            horizon: Forecast horizon

        Returns:
            Dictionary with EWMA volatility forecast
        """
        returns = self.df['Returns'].dropna()

        # Calculate EWMA variance
        ewma_var = returns.ewm(alpha=1-lambda_param).var()
        current_ewma_vol = np.sqrt(ewma_var.iloc[-1])

        # EWMA forecast (assumes constant volatility)
        forecast = [float(current_ewma_vol)] * horizon

        return {
            'method': 'EWMA',
            'lambda': lambda_param,
            'current_volatility': float(current_ewma_vol),
            'forecasts': forecast,
            'mean_forecast': float(current_ewma_vol)
        }

    def detect_volatility_regime(self) -> Dict[str, any]:
        """
        Detect current volatility regime (low, normal, high)

        Returns:
            Dictionary with regime information
        """
        returns = self.df['Returns'].dropna()

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()

        # Current volatility
        current_vol = rolling_vol.iloc[-1]

        # Historical percentiles
        vol_25 = rolling_vol.quantile(0.25)
        vol_50 = rolling_vol.quantile(0.50)
        vol_75 = rolling_vol.quantile(0.75)

        # Determine regime
        if current_vol < vol_25:
            regime = 'Low Volatility'
            interpretation = 'Market is calm, expect potential volatility expansion'
        elif current_vol < vol_75:
            regime = 'Normal Volatility'
            interpretation = 'Market volatility is within normal range'
        else:
            regime = 'High Volatility'
            interpretation = 'Market is volatile, heightened risk'

        # Volatility trend
        recent_vol_trend = rolling_vol.iloc[-10:].values
        vol_slope = np.polyfit(range(len(recent_vol_trend)), recent_vol_trend, 1)[0]

        if vol_slope > 0.1:
            trend = 'Increasing'
        elif vol_slope < -0.1:
            trend = 'Decreasing'
        else:
            trend = 'Stable'

        return {
            'current_volatility': float(current_vol),
            'regime': regime,
            'trend': trend,
            'interpretation': interpretation,
            'percentile': float(stats.percentileofscore(rolling_vol.dropna(), current_vol)),
            'historical_stats': {
                '25th_percentile': float(vol_25),
                'median': float(vol_50),
                '75th_percentile': float(vol_75),
                'mean': float(rolling_vol.mean()),
                'std': float(rolling_vol.std())
            }
        }

    def detect_volatility_clustering(self, window: int = 50) -> Dict[str, any]:
        """
        Detect volatility clustering patterns

        Args:
            window: Lookback window for analysis

        Returns:
            Dictionary with clustering information
        """
        returns = self.df['Returns'].dropna().iloc[-window:]

        # Calculate squared returns (proxy for volatility)
        squared_returns = returns ** 2

        # Autocorrelation of squared returns (indicates clustering)
        autocorr_lags = [1, 5, 10]
        autocorrelations = {}

        for lag in autocorr_lags:
            autocorr = squared_returns.autocorr(lag=lag)
            autocorrelations[f'lag_{lag}'] = float(autocorr)

        # High autocorrelation indicates clustering
        avg_autocorr = np.mean(list(autocorrelations.values()))

        if avg_autocorr > 0.3:
            clustering = 'Strong'
            interpretation = 'High volatility tends to follow high volatility'
        elif avg_autocorr > 0.1:
            clustering = 'Moderate'
            interpretation = 'Some volatility persistence observed'
        else:
            clustering = 'Weak'
            interpretation = 'Little evidence of volatility clustering'

        return {
            'clustering_strength': clustering,
            'interpretation': interpretation,
            'autocorrelations': autocorrelations,
            'average_autocorr': float(avg_autocorr)
        }

    def calculate_breakout_probability(self, lookback: int = 20) -> Dict[str, any]:
        """
        Calculate probability of breakout from current consolidation range

        Args:
            lookback: Lookback period for range definition

        Returns:
            Dictionary with breakout analysis
        """
        recent_data = self.df.iloc[-lookback:]

        # Define consolidation range
        high_range = recent_data['High'].max()
        low_range = recent_data['Low'].min()
        range_size = high_range - low_range
        current_price = self.df['Close'].iloc[-1]

        # Calculate position in range
        position_in_range = (current_price - low_range) / range_size if range_size > 0 else 0.5

        # Calculate volatility metrics
        returns = self.df['Returns'].dropna().iloc[-lookback:]
        current_vol = returns.std()
        avg_vol = self.df['Returns'].dropna().iloc[-100:].std()

        # Volatility ratio
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Bollinger Bands squeeze indicator
        bb_width = (high_range - low_range) / recent_data['Close'].mean()
        historical_bb_width = []

        for i in range(len(self.df) - lookback):
            if i >= lookback:
                window = self.df.iloc[i-lookback:i]
                width = (window['High'].max() - window['Low'].min()) / window['Close'].mean()
                historical_bb_width.append(width)

        if historical_bb_width:
            bb_percentile = stats.percentileofscore(historical_bb_width, bb_width)
        else:
            bb_percentile = 50

        # Breakout probability based on multiple factors
        squeeze_score = (100 - bb_percentile) / 100  # Lower width = higher squeeze
        vol_contraction_score = max(0, (1 - vol_ratio))  # Lower vol = higher score

        # Combined probability
        breakout_prob = (squeeze_score * 0.6 + vol_contraction_score * 0.4) * 100

        # Direction bias
        if position_in_range > 0.6:
            direction_bias = 'Upward'
        elif position_in_range < 0.4:
            direction_bias = 'Downward'
        else:
            direction_bias = 'Neutral'

        return {
            'breakout_probability_%': float(breakout_prob),
            'direction_bias': direction_bias,
            'consolidation_range': {
                'high': float(high_range),
                'low': float(low_range),
                'width_%': float(bb_width * 100)
            },
            'current_price': float(current_price),
            'position_in_range_%': float(position_in_range * 100),
            'volatility_status': 'Contracting' if vol_ratio < 0.8 else 'Normal' if vol_ratio < 1.2 else 'Expanding',
            'squeeze_indicator': 'Tight' if bb_percentile < 20 else 'Normal' if bb_percentile < 80 else 'Wide'
        }

    def calculate_var(self, confidence_level: float = 0.95, horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR)

        Args:
            confidence_level: Confidence level (default 0.95 for 95%)
            horizon: Time horizon in periods

        Returns:
            Dictionary with VaR metrics
        """
        returns = self.df['Returns'].dropna()
        current_price = self.df['Close'].iloc[-1]

        # Historical VaR
        var_percentile = (1 - confidence_level) * 100
        historical_var = np.percentile(returns, var_percentile)

        # Parametric VaR (assumes normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        parametric_var = mean_return + z_score * std_return

        # Scale to horizon
        horizon_multiplier = np.sqrt(horizon)
        historical_var_horizon = historical_var * horizon_multiplier
        parametric_var_horizon = parametric_var * horizon_multiplier

        # Convert to dollar value
        historical_var_dollar = current_price * (historical_var_horizon / 100)
        parametric_var_dollar = current_price * (parametric_var_horizon / 100)

        return {
            'confidence_level_%': confidence_level * 100,
            'horizon_periods': horizon,
            'current_price': float(current_price),
            'historical_var_%': float(historical_var_horizon),
            'historical_var_$': float(historical_var_dollar),
            'parametric_var_%': float(parametric_var_horizon),
            'parametric_var_$': float(parametric_var_dollar),
            'interpretation': f'With {confidence_level*100}% confidence, loss will not exceed {abs(historical_var_horizon):.2f}% over {horizon} period(s)'
        }

    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Expected Shortfall (Conditional VaR)

        Args:
            confidence_level: Confidence level

        Returns:
            Dictionary with ES metrics
        """
        returns = self.df['Returns'].dropna()
        current_price = self.df['Close'].iloc[-1]

        # Calculate VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)

        # Expected Shortfall: average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        es = tail_returns.mean()

        # Convert to dollar value
        es_dollar = current_price * (es / 100)

        return {
            'confidence_level_%': confidence_level * 100,
            'current_price': float(current_price),
            'expected_shortfall_%': float(es),
            'expected_shortfall_$': float(es_dollar),
            'var_threshold_%': float(var_threshold),
            'tail_observations': len(tail_returns),
            'interpretation': f'Given a loss event (worst {(1-confidence_level)*100}%), expected loss is {abs(es):.2f}%'
        }

    def calculate_historical_volatility(self, windows: List[int] = [10, 20, 30, 60, 90]) -> Dict[str, float]:
        """
        Calculate historical volatility for multiple time windows

        Args:
            windows: List of window sizes in periods

        Returns:
            Dictionary with volatility for each window
        """
        returns = self.df['Returns'].dropna()

        result = {
            'current_price': float(self.df['Close'].iloc[-1]),
            'windows': {}
        }

        for window in windows:
            if len(returns) >= window:
                vol = returns.iloc[-window:].std()
                # Annualized volatility (assuming 252 trading days)
                annualized_vol = vol * np.sqrt(252)

                result['windows'][f'{window}_day'] = {
                    'volatility_%': float(vol),
                    'annualized_%': float(annualized_vol),
                    'mean_return_%': float(returns.iloc[-window:].mean())
                }

        return result

    def calculate_parkinson_volatility(self, window: int = 20) -> Dict[str, float]:
        """
        Parkinson's volatility estimator using high-low range
        More efficient than close-to-close volatility

        Args:
            window: Lookback window

        Returns:
            Dictionary with Parkinson volatility
        """
        high = self.df['High']
        low = self.df['Low']

        # Parkinson formula: sqrt(1/(4*ln(2)) * mean((ln(H/L))^2))
        hl_ratio = np.log(high / low)
        parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)

        # Rolling calculation
        parkinson_vol = np.sqrt(parkinson_var.rolling(window=window).mean()) * 100
        current_vol = parkinson_vol.iloc[-1]

        # Annualized
        annualized = current_vol * np.sqrt(252)

        return {
            'method': 'Parkinson High-Low',
            'window': window,
            'volatility_%': float(current_vol),
            'annualized_%': float(annualized),
            'interpretation': 'Uses high-low range, more efficient than close-to-close'
        }

    def calculate_garman_klass_volatility(self, window: int = 20) -> Dict[str, float]:
        """
        Garman-Klass volatility estimator using OHLC
        More efficient than Parkinson, accounts for opening jumps

        Args:
            window: Lookback window

        Returns:
            Dictionary with Garman-Klass volatility
        """
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        open_price = self.df['Open']

        # Garman-Klass formula
        hl_component = 0.5 * (np.log(high / low) ** 2)
        co_component = (2 * np.log(2) - 1) * (np.log(close / open_price) ** 2)

        gk_var = hl_component - co_component
        gk_vol = np.sqrt(gk_var.rolling(window=window).mean()) * 100
        current_vol = gk_vol.iloc[-1]

        # Annualized
        annualized = current_vol * np.sqrt(252)

        return {
            'method': 'Garman-Klass OHLC',
            'window': window,
            'volatility_%': float(current_vol),
            'annualized_%': float(annualized),
            'interpretation': 'Uses OHLC data, accounts for opening jumps'
        }

    def calculate_rogers_satchell_volatility(self, window: int = 20) -> Dict[str, float]:
        """
        Rogers-Satchell volatility estimator
        Allows for drift, more accurate for trending markets

        Args:
            window: Lookback window

        Returns:
            Dictionary with Rogers-Satchell volatility
        """
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        open_price = self.df['Open']

        # Rogers-Satchell formula
        rs_var = (np.log(high / close) * np.log(high / open_price) +
                  np.log(low / close) * np.log(low / open_price))

        rs_vol = np.sqrt(rs_var.rolling(window=window).mean()) * 100
        current_vol = rs_vol.iloc[-1]

        # Annualized
        annualized = current_vol * np.sqrt(252)

        return {
            'method': 'Rogers-Satchell',
            'window': window,
            'volatility_%': float(current_vol),
            'annualized_%': float(annualized),
            'interpretation': 'Drift-independent estimator, better for trending markets'
        }

    def calculate_yang_zhang_volatility(self, window: int = 20) -> Dict[str, float]:
        """
        Yang-Zhang volatility estimator
        Most efficient estimator combining overnight and intraday volatility

        Args:
            window: Lookback window

        Returns:
            Dictionary with Yang-Zhang volatility
        """
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        open_price = self.df['Open']

        # Overnight volatility
        overnight_ret = np.log(open_price / close.shift(1))
        overnight_var = overnight_ret.rolling(window=window).var()

        # Opening volatility
        open_to_close = np.log(close / open_price)
        open_var = open_to_close.rolling(window=window).var()

        # Rogers-Satchell component
        rs_var = (np.log(high / close) * np.log(high / open_price) +
                  np.log(low / close) * np.log(low / open_price))
        rs_var_roll = rs_var.rolling(window=window).mean()

        # Yang-Zhang formula with k parameter
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        yz_var = overnight_var + k * open_var + (1 - k) * rs_var_roll
        yz_vol = np.sqrt(yz_var) * 100
        current_vol = yz_vol.iloc[-1]

        # Annualized
        annualized = current_vol * np.sqrt(252)

        return {
            'method': 'Yang-Zhang',
            'window': window,
            'volatility_%': float(current_vol),
            'annualized_%': float(annualized),
            'interpretation': 'Most efficient estimator, combines overnight and intraday volatility'
        }

    def calculate_realized_volatility(self, window: int = 20) -> Dict[str, float]:
        """
        Calculate realized volatility (sum of squared returns)

        Args:
            window: Lookback window

        Returns:
            Dictionary with realized volatility
        """
        returns = self.df['Returns'].dropna()

        # Realized variance = sum of squared returns
        realized_var = (returns ** 2).rolling(window=window).sum()
        realized_vol = np.sqrt(realized_var)

        current_vol = realized_vol.iloc[-1]
        annualized = current_vol * np.sqrt(252 / window)

        return {
            'method': 'Realized Volatility',
            'window': window,
            'volatility_%': float(current_vol),
            'annualized_%': float(annualized),
            'interpretation': 'Sum of squared returns, measures actual price variation'
        }

    def calculate_volatility_cones(self, windows: List[int] = [10, 20, 30, 60, 90]) -> pd.DataFrame:
        """
        Calculate volatility cones (percentile bands) for different time windows

        Args:
            windows: List of window sizes

        Returns:
            DataFrame with volatility statistics for each window
        """
        returns = self.df['Returns'].dropna()

        cone_data = []

        for window in windows:
            if len(returns) < window * 3:  # Need enough data
                continue

            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=window).std()
            rolling_vol = rolling_vol.dropna()

            if len(rolling_vol) > 0:
                cone_data.append({
                    'Window': f'{window}d',
                    'Current_%': float(rolling_vol.iloc[-1]),
                    'Min_%': float(rolling_vol.min()),
                    '10th_%': float(rolling_vol.quantile(0.10)),
                    '25th_%': float(rolling_vol.quantile(0.25)),
                    'Median_%': float(rolling_vol.median()),
                    '75th_%': float(rolling_vol.quantile(0.75)),
                    '90th_%': float(rolling_vol.quantile(0.90)),
                    'Max_%': float(rolling_vol.max()),
                    'Mean_%': float(rolling_vol.mean()),
                    'Percentile': float(stats.percentileofscore(rolling_vol, rolling_vol.iloc[-1]))
                })

        return pd.DataFrame(cone_data)

    def calculate_volatility_ratios(self) -> Dict[str, float]:
        """
        Calculate various volatility ratios for comparison

        Returns:
            Dictionary with volatility ratios
        """
        returns = self.df['Returns'].dropna()

        # Short-term vs long-term
        vol_10d = returns.iloc[-10:].std()
        vol_20d = returns.iloc[-20:].std()
        vol_60d = returns.iloc[-60:].std() if len(returns) >= 60 else vol_20d

        # Parkinson vs Close-to-Close
        parkinson = self.calculate_parkinson_volatility(window=20)

        return {
            'short_to_long_ratio': float(vol_10d / vol_60d) if vol_60d > 0 else 1.0,
            'vol_10d_to_20d': float(vol_10d / vol_20d) if vol_20d > 0 else 1.0,
            'parkinson_to_close': float(parkinson['volatility_%'] / vol_20d) if vol_20d > 0 else 1.0,
            'current_10d_%': float(vol_10d),
            'current_20d_%': float(vol_20d),
            'current_60d_%': float(vol_60d),
            'interpretation': {
                'short_to_long': 'Rising' if vol_10d / vol_60d > 1.2 else 'Falling' if vol_10d / vol_60d < 0.8 else 'Stable'
            }
        }

    def calculate_volatility_skew(self, window: int = 60) -> Dict[str, float]:
        """
        Analyze volatility skew (asymmetry in volatility response to up/down moves)

        Args:
            window: Lookback window

        Returns:
            Dictionary with skew analysis
        """
        recent_data = self.df.iloc[-window:]
        returns = recent_data['Returns'].dropna()

        # Separate positive and negative returns
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        # Calculate volatility for each
        vol_up = positive_returns.std() if len(positive_returns) > 5 else 0
        vol_down = negative_returns.std() if len(negative_returns) > 5 else 0

        # Skew ratio
        skew_ratio = vol_down / vol_up if vol_up > 0 else 1.0

        # Return skewness
        returns_skew = returns.skew()

        return {
            'volatility_up_days_%': float(vol_up),
            'volatility_down_days_%': float(vol_down),
            'skew_ratio': float(skew_ratio),
            'returns_skewness': float(returns_skew),
            'interpretation': 'Negative skew' if skew_ratio > 1.1 else 'Positive skew' if skew_ratio < 0.9 else 'Symmetric',
            'meaning': 'Down moves more volatile' if skew_ratio > 1.1 else 'Up moves more volatile' if skew_ratio < 0.9 else 'Balanced volatility'
        }

    def calculate_volatility_term_structure(self) -> pd.DataFrame:
        """
        Calculate volatility term structure across different horizons

        Returns:
            DataFrame with term structure
        """
        returns = self.df['Returns'].dropna()

        horizons = [5, 10, 20, 30, 60, 90]
        term_structure = []

        for horizon in horizons:
            if len(returns) >= horizon:
                vol = returns.iloc[-horizon:].std()
                annualized = vol * np.sqrt(252)

                term_structure.append({
                    'Horizon_Days': horizon,
                    'Volatility_%': float(vol),
                    'Annualized_%': float(annualized)
                })

        df = pd.DataFrame(term_structure)

        # Add slope
        if len(df) >= 2:
            slope = (df['Volatility_%'].iloc[-1] - df['Volatility_%'].iloc[0]) / (df['Horizon_Days'].iloc[-1] - df['Horizon_Days'].iloc[0])
            df.attrs['slope'] = float(slope)
            df.attrs['shape'] = 'Upward' if slope > 0.05 else 'Downward' if slope < -0.05 else 'Flat'

        return df

    def get_all_volatility_measurements(self) -> Dict[str, any]:
        """
        Get all volatility measurements in one comprehensive report

        Returns:
            Dictionary with all volatility metrics
        """
        report = {
            'historical_volatility': self.calculate_historical_volatility(),
            'parkinson': self.calculate_parkinson_volatility(),
            'garman_klass': self.calculate_garman_klass_volatility(),
            'rogers_satchell': self.calculate_rogers_satchell_volatility(),
            'yang_zhang': self.calculate_yang_zhang_volatility(),
            'realized': self.calculate_realized_volatility(),
            'volatility_cones': self.calculate_volatility_cones().to_dict('records'),
            'volatility_ratios': self.calculate_volatility_ratios(),
            'volatility_skew': self.calculate_volatility_skew(),
            'term_structure': self.calculate_volatility_term_structure().to_dict('records')
        }

        return report

    def compare_volatility_models(self) -> pd.DataFrame:
        """
        Compare different volatility forecasting models

        Returns:
            DataFrame comparing model forecasts
        """
        garch = self.predict_volatility_garch(horizon=1)
        ewma = self.predict_volatility_ewma(horizon=1)

        # Simple historical
        returns = self.df['Returns'].dropna()
        hist_vol = returns.iloc[-20:].std()

        comparison = pd.DataFrame([
            {
                'Model': 'GARCH(1,1)',
                'Next_Period_Volatility_%': garch['forecasts'][0],
                'Method': 'Time series model with volatility clustering'
            },
            {
                'Model': 'EWMA',
                'Next_Period_Volatility_%': ewma['forecasts'][0],
                'Method': 'Exponentially weighted moving average'
            },
            {
                'Model': 'Historical (20-day)',
                'Next_Period_Volatility_%': float(hist_vol),
                'Method': 'Simple historical standard deviation'
            }
        ])

        return comparison

    def get_comprehensive_volatility_report(self) -> Dict[str, any]:
        """
        Get comprehensive volatility analysis report

        Returns:
            Dictionary with all volatility metrics
        """
        report = {
            # All measurements
            'measurements': self.get_all_volatility_measurements(),
            # Predictions and forecasts
            'garch_forecast': self.predict_volatility_garch(horizon=5),
            'ewma_forecast': self.predict_volatility_ewma(horizon=5),
            'volatility_regime': self.detect_volatility_regime(),
            'volatility_clustering': self.detect_volatility_clustering(),
            'breakout_analysis': self.calculate_breakout_probability(),
            'var_95': self.calculate_var(confidence_level=0.95, horizon=1),
            'var_99': self.calculate_var(confidence_level=0.99, horizon=1),
            'expected_shortfall': self.calculate_expected_shortfall(confidence_level=0.95)
        }

        return report
