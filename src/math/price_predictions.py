"""
Price Predictions Module
- Next period's price (close, high, low)
- Price range forecasts
- Multi-step ahead price predictions
- Expected price at specific future dates
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PricePredictor:
    """Comprehensive price prediction using statistical methods"""

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

    def predict_next_price(self, method: str = 'all') -> Dict[str, float]:
        """
        Predict next period's price using various methods

        Args:
            method: Prediction method ('linear', 'arima', 'exponential', 'momentum', 'all')

        Returns:
            Dictionary with predicted prices for close, high, low
        """
        predictions = {}

        if method in ['linear', 'all']:
            predictions['linear'] = self._predict_linear_regression()

        if method in ['arima', 'all']:
            predictions['arima'] = self._predict_arima()

        if method in ['exponential', 'all']:
            predictions['exponential'] = self._predict_exponential_smoothing()

        if method in ['momentum', 'all']:
            predictions['momentum'] = self._predict_momentum_based()

        if method == 'all':
            # Ensemble: average of all methods
            close_predictions = [p['close'] for p in predictions.values() if 'close' in p]
            high_predictions = [p['high'] for p in predictions.values() if 'high' in p]
            low_predictions = [p['low'] for p in predictions.values() if 'low' in p]

            predictions['ensemble'] = {
                'close': float(np.mean(close_predictions)),
                'high': float(np.mean(high_predictions)),
                'low': float(np.mean(low_predictions))
            }

        return predictions

    def _predict_linear_regression(self, window: int = 50) -> Dict[str, float]:
        """Predict using linear regression on recent data"""
        recent_data = self.df.iloc[-window:]
        X = np.arange(len(recent_data)).reshape(-1, 1)
        next_X = np.array([[len(recent_data)]])

        predictions = {}
        for col in ['Close', 'High', 'Low']:
            y = recent_data[col].values
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(next_X)[0]
            predictions[col.lower()] = float(pred)

        return predictions

    def _predict_arima(self, order: Tuple[int, int, int] = (2, 1, 2)) -> Dict[str, float]:
        """Predict using ARIMA model"""
        predictions = {}

        try:
            # Use last 100 points for faster computation
            recent_close = self.df['Close'].iloc[-100:]

            model = ARIMA(recent_close, order=order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=1)
            close_pred = float(forecast.iloc[0])

            # Estimate high/low based on recent volatility
            recent_volatility = recent_close.pct_change().std()
            predictions['close'] = close_pred
            predictions['high'] = close_pred * (1 + recent_volatility)
            predictions['low'] = close_pred * (1 - recent_volatility)

        except Exception as e:
            # Fallback to simple moving average
            predictions['close'] = float(self.df['Close'].iloc[-20:].mean())
            predictions['high'] = float(self.df['High'].iloc[-20:].mean())
            predictions['low'] = float(self.df['Low'].iloc[-20:].mean())

        return predictions

    def _predict_exponential_smoothing(self) -> Dict[str, float]:
        """Predict using exponential smoothing"""
        predictions = {}

        try:
            # Use last 100 points
            recent_close = self.df['Close'].iloc[-100:]

            # Simple Exponential Smoothing
            model = ExponentialSmoothing(recent_close, trend='add', seasonal=None)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=1)
            close_pred = float(forecast.iloc[0])

            # Estimate high/low
            recent_volatility = recent_close.pct_change().std()
            predictions['close'] = close_pred
            predictions['high'] = close_pred * (1 + recent_volatility)
            predictions['low'] = close_pred * (1 - recent_volatility)

        except Exception as e:
            # Fallback to exponential weighted moving average
            alpha = 0.3
            predictions['close'] = float(self.df['Close'].ewm(alpha=alpha).mean().iloc[-1])
            predictions['high'] = float(self.df['High'].ewm(alpha=alpha).mean().iloc[-1])
            predictions['low'] = float(self.df['Low'].ewm(alpha=alpha).mean().iloc[-1])

        return predictions

    def _predict_momentum_based(self, window: int = 20) -> Dict[str, float]:
        """Predict based on momentum and trend continuation"""
        recent_data = self.df.iloc[-window:]

        # Calculate momentum
        momentum = recent_data['Close'].pct_change().mean()
        current_close = self.df['Close'].iloc[-1]

        # Project based on average momentum
        predictions = {
            'close': float(current_close * (1 + momentum)),
            'high': float(self.df['High'].iloc[-1] * (1 + momentum)),
            'low': float(self.df['Low'].iloc[-1] * (1 + momentum))
        }

        return predictions

    def predict_price_range(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Predict price range with confidence interval

        Args:
            confidence: Confidence level (default 0.95 for 95%)

        Returns:
            Dictionary with (lower_bound, upper_bound) for each price type
        """
        z_score = stats.norm.ppf((1 + confidence) / 2)
        ranges = {}

        for col in ['Close', 'High', 'Low']:
            # Calculate recent volatility
            returns = self.df[col].pct_change().dropna()
            std_dev = returns.std()
            mean_return = returns.mean()

            current_price = self.df[col].iloc[-1]

            # Expected price
            expected = current_price * (1 + mean_return)

            # Confidence interval
            margin = current_price * std_dev * z_score
            lower = expected - margin
            upper = expected + margin

            ranges[col.lower()] = (float(lower), float(upper))

        return ranges

    def predict_multi_step(self, steps: int = 5, method: str = 'exponential') -> pd.DataFrame:
        """
        Multi-step ahead predictions

        Args:
            steps: Number of periods to predict ahead
            method: Prediction method to use

        Returns:
            DataFrame with predictions for multiple steps
        """
        predictions = []

        if method == 'exponential':
            try:
                # Use exponential smoothing for multi-step
                recent_close = self.df['Close'].iloc[-100:]
                model = ExponentialSmoothing(recent_close, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=steps)

                for i, close_pred in enumerate(forecast, 1):
                    recent_volatility = recent_close.pct_change().std()
                    predictions.append({
                        'Step': i,
                        'Predicted_Close': float(close_pred),
                        'Predicted_High': float(close_pred * (1 + recent_volatility)),
                        'Predicted_Low': float(close_pred * (1 - recent_volatility))
                    })

            except Exception as e:
                # Fallback to simple trend continuation
                return self._predict_multi_step_simple(steps)

        elif method == 'arima':
            try:
                recent_close = self.df['Close'].iloc[-100:]
                model = ARIMA(recent_close, order=(2, 1, 2))
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=steps)

                recent_volatility = recent_close.pct_change().std()
                for i, close_pred in enumerate(forecast, 1):
                    predictions.append({
                        'Step': i,
                        'Predicted_Close': float(close_pred),
                        'Predicted_High': float(close_pred * (1 + recent_volatility)),
                        'Predicted_Low': float(close_pred * (1 - recent_volatility))
                    })

            except Exception as e:
                return self._predict_multi_step_simple(steps)

        else:
            return self._predict_multi_step_simple(steps)

        return pd.DataFrame(predictions)

    def _predict_multi_step_simple(self, steps: int) -> pd.DataFrame:
        """Simple multi-step prediction fallback"""
        predictions = []
        momentum = self.df['Close'].pct_change().mean()
        current_close = self.df['Close'].iloc[-1]

        for i in range(1, steps + 1):
            predicted_close = current_close * ((1 + momentum) ** i)
            volatility = self.df['Close'].pct_change().std()

            predictions.append({
                'Step': i,
                'Predicted_Close': float(predicted_close),
                'Predicted_High': float(predicted_close * (1 + volatility)),
                'Predicted_Low': float(predicted_close * (1 - volatility))
            })

        return pd.DataFrame(predictions)

    def predict_target_date(self, target_date: str) -> Dict[str, float]:
        """
        Predict price at a specific future date

        Args:
            target_date: Target date in YYYY-MM-DD format

        Returns:
            Dictionary with predicted prices
        """
        target = pd.to_datetime(target_date)
        last_date = self.df.index[-1]

        # Calculate number of trading days (approximate)
        days_diff = (target - last_date).days
        trading_days = int(days_diff * (5/7))  # Approximate trading days

        if trading_days <= 0:
            return {'error': 'Target date must be in the future'}

        # Use exponential smoothing for longer forecasts
        try:
            recent_close = self.df['Close'].iloc[-100:]
            model = ExponentialSmoothing(recent_close, trend='add', seasonal=None)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=trading_days)

            close_pred = float(forecast.iloc[-1])
            recent_volatility = recent_close.pct_change().std() * np.sqrt(trading_days)

            return {
                'target_date': target_date,
                'trading_days_ahead': trading_days,
                'predicted_close': close_pred,
                'confidence_range_low': float(close_pred * (1 - recent_volatility)),
                'confidence_range_high': float(close_pred * (1 + recent_volatility)),
                'current_price': float(self.df['Close'].iloc[-1]),
                'expected_change_%': float((close_pred / self.df['Close'].iloc[-1] - 1) * 100)
            }

        except Exception as e:
            # Simple momentum-based prediction
            momentum = self.df['Close'].pct_change().mean()
            current_close = self.df['Close'].iloc[-1]
            predicted = current_close * ((1 + momentum) ** trading_days)

            return {
                'target_date': target_date,
                'trading_days_ahead': trading_days,
                'predicted_close': float(predicted),
                'current_price': float(current_close),
                'expected_change_%': float((predicted / current_close - 1) * 100)
            }

    def get_prediction_summary(self) -> pd.DataFrame:
        """
        Get a comprehensive summary of next period predictions

        Returns:
            DataFrame with predictions from all methods
        """
        all_predictions = self.predict_next_price(method='all')

        summary_data = []
        for method, preds in all_predictions.items():
            summary_data.append({
                'Method': method.capitalize(),
                'Predicted_Close': f"${preds['close']:.2f}",
                'Predicted_High': f"${preds['high']:.2f}",
                'Predicted_Low': f"${preds['low']:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)

        # Add current prices for reference
        current = {
            'Method': 'Current',
            'Predicted_Close': f"${self.df['Close'].iloc[-1]:.2f}",
            'Predicted_High': f"${self.df['High'].iloc[-1]:.2f}",
            'Predicted_Low': f"${self.df['Low'].iloc[-1]:.2f}"
        }

        summary_df = pd.concat([pd.DataFrame([current]), summary_df], ignore_index=True)

        return summary_df

    def calculate_prediction_accuracy(self, test_periods: int = 20) -> Dict[str, float]:
        """
        Calculate historical prediction accuracy

        Args:
            test_periods: Number of periods to backtest

        Returns:
            Dictionary with accuracy metrics for each method
        """
        if len(self.df) < test_periods + 50:
            return {'error': 'Insufficient data for accuracy calculation'}

        # Store original dataframe
        original_df = self.df.copy()

        errors = {'linear': [], 'momentum': [], 'exponential': []}

        for i in range(test_periods):
            # Use data up to -i-1 to predict -i
            self.df = original_df.iloc[:-(i+1)]

            actual_price = original_df['Close'].iloc[-(i+1)]

            # Get predictions
            pred_linear = self._predict_linear_regression()
            pred_momentum = self._predict_momentum_based()
            pred_exp = self._predict_exponential_smoothing()

            # Calculate errors
            errors['linear'].append(abs(pred_linear['close'] - actual_price) / actual_price)
            errors['momentum'].append(abs(pred_momentum['close'] - actual_price) / actual_price)
            errors['exponential'].append(abs(pred_exp['close'] - actual_price) / actual_price)

        # Restore original dataframe
        self.df = original_df

        # Calculate mean absolute percentage error
        accuracy = {
            'linear_mape': float(np.mean(errors['linear']) * 100),
            'momentum_mape': float(np.mean(errors['momentum']) * 100),
            'exponential_mape': float(np.mean(errors['exponential']) * 100),
            'test_periods': test_periods
        }

        return accuracy

