"""
Mathematical & Statistical Predictions
Price Predictions:

Next period's price (close, high, low)
Price range forecasts
Multi-step ahead price predictions
Expected price at specific future dates

Trend Analysis:

Trend direction (uptrend, downtrend, sideways)
Trend strength and momentum
Trend reversal points
Cycle identification (periodicity in price movements)

Volatility Predictions:

Future volatility estimates (GARCH models)
Volatility clustering patterns
Breakout probability from consolidation ranges
Risk metrics (Value at Risk, Expected Shortfall)

Pattern Recognition:

Similar historical pattern matching
Candlestick pattern completion probability
Chart pattern breakout/breakdown predictions
Support/resistance level strength
"""

import sys
import json
from pathlib import Path

# Import all analysis modules
from price_predictions import PricePredictor
from trend_analysis import TrendAnalyzer
from volatility_predictions import VolatilityPredictor
from pattern_recognition import PatternRecognizer


def list_csv_files(data_dir: Path) -> list:
    """
    List all CSV files in the data directory

    Args:
        data_dir: Path to data directory

    Returns:
        List of CSV file paths
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    return csv_files


def select_data_file() -> str:
    """
    Interactive CSV file selection from data directory

    Returns:
        Path to selected CSV file
    """
    # Determine data directory path (relative to this script)
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../../data"
    data_dir = data_dir.resolve()

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Get all CSV files
    csv_files = list_csv_files(data_dir)

    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        sys.exit(1)

    # Display available files
    print("\n" + "=" * 80)
    print("AVAILABLE DATA FILES".center(80))
    print("=" * 80)
    print(f"\nFound {len(csv_files)} CSV file(s) in: {data_dir}\n")

    for idx, file_path in enumerate(csv_files, 1):
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"  [{idx}] {file_path.name:<30} ({file_size:.1f} KB)")

    print("\n" + "=" * 80)

    # Get user selection
    while True:
        try:
            choice = input(f"\nSelect a file [1-{len(csv_files)}] or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)

            choice_num = int(choice)

            if 1 <= choice_num <= len(csv_files):
                selected_file = csv_files[choice_num - 1]
                print(f"\n✓ Selected: {selected_file.name}")
                return str(selected_file)
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")

        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def run_complete_analysis(data_path: str, export_json: bool = False):
    """
    Run complete mathematical and statistical analysis

    Args:
        data_path: Path to OHLCV CSV file
        export_json: Whether to export results as JSON
    """
    print_section("MATHEMATICAL & STATISTICAL MARKET ANALYSIS")
    print(f"\nAnalyzing: {Path(data_path).name}")

    results = {}

    # ============================================================================
    # PRICE PREDICTIONS
    # ============================================================================
    print_section("1. PRICE PREDICTIONS")

    try:
        predictor = PricePredictor(data_path)

        print("\n--- Next Period Predictions ---")
        summary = predictor.get_prediction_summary()
        print(summary.to_string(index=False))

        print("\n--- Price Range Forecasts (95% Confidence) ---")
        ranges = predictor.predict_price_range()
        for target, (lower, upper) in ranges.items():
            current = predictor.df[target.capitalize()].iloc[-1]
            print(f"\n{target.capitalize()}:")
            print(f"  Current: ${current:.2f}")
            print(f"  Range: ${lower:.2f} - ${upper:.2f}")

        print("\n--- Multi-Step Predictions (Next 5 Periods) ---")
        multi_step = predictor.predict_multi_step(steps=5)
        print(multi_step.to_string(index=False))

        results['price_predictions'] = {
            'next_period': predictor.predict_next_price(method='all'),
            'price_ranges': ranges,
            'multi_step': multi_step.to_dict('records')
        }

        print("\n✓ Price predictions completed successfully")

    except Exception as e:
        print(f"\n✗ Error in price predictions: {e}")
        results['price_predictions'] = {'error': str(e)}

    # ============================================================================
    # TREND ANALYSIS
    # ============================================================================
    print_section("2. TREND ANALYSIS")

    try:
        analyzer = TrendAnalyzer(data_path)

        print("\n--- Trend Direction ---")
        direction = analyzer.detect_trend_direction()
        print(f"Current Trend: {direction['trend']}")
        print(f"Confidence: {direction['confidence']:.1%}")
        print(f"R-Squared: {direction['r_squared']:.4f}")
        print(f"Price Change: {direction['price_change_%']:.2f}%")

        print("\n--- Trend Strength ---")
        strength = analyzer.calculate_trend_strength()
        print(f"ADX: {strength['adx']:.2f}")
        print(f"Trend Strength: {strength['trend_strength']}")
        print(f"Interpretation: {strength['interpretation']}")

        print("\n--- Momentum Analysis ---")
        momentum = analyzer.detect_momentum()
        print(momentum.to_string(index=False))

        print("\n--- Reversal Point Detection ---")
        reversal = analyzer.detect_reversal_points()
        print(f"Reversal Probability: {reversal['reversal_probability']}")
        print(f"RSI: {reversal['rsi']:.2f}")
        if reversal['signals']:
            print("Signals:")
            for signal in reversal['signals']:
                print(f"  • {signal}")

        print("\n--- Cycle Identification ---")
        cycles = analyzer.detect_cycles()
        if cycles['fft_cycles']:
            print("FFT-Based Cycles:")
            for i, cycle in enumerate(cycles['fft_cycles'][:3], 1):
                print(f"  {i}. Period: {cycle['period_days']:.1f} days")

        results['trend_analysis'] = {
            'direction': direction,
            'strength': strength,
            'momentum': momentum.to_dict('records'),
            'reversal': reversal,
            'cycles': cycles
        }

        print("\n✓ Trend analysis completed successfully")

    except Exception as e:
        print(f"\n✗ Error in trend analysis: {e}")
        results['trend_analysis'] = {'error': str(e)}

    # ============================================================================
    # VOLATILITY PREDICTIONS
    # ============================================================================
    print_section("3. VOLATILITY PREDICTIONS")

    try:
        vol_predictor = VolatilityPredictor(data_path)

        print("\n--- Historical Volatility (Multiple Windows) ---")
        hist_vol = vol_predictor.calculate_historical_volatility()
        for window, metrics in hist_vol['windows'].items():
            print(f"{window}: {metrics['volatility_%']:.2f}% (Ann: {metrics['annualized_%']:.1f}%)")

        print("\n--- Advanced Volatility Estimators ---")
        parkinson = vol_predictor.calculate_parkinson_volatility()
        print(f"Parkinson (High-Low): {parkinson['volatility_%']:.2f}% (Ann: {parkinson['annualized_%']:.1f}%)")

        gk = vol_predictor.calculate_garman_klass_volatility()
        print(f"Garman-Klass (OHLC): {gk['volatility_%']:.2f}% (Ann: {gk['annualized_%']:.1f}%)")

        rs = vol_predictor.calculate_rogers_satchell_volatility()
        print(f"Rogers-Satchell: {rs['volatility_%']:.2f}% (Ann: {rs['annualized_%']:.1f}%)")

        yz = vol_predictor.calculate_yang_zhang_volatility()
        print(f"Yang-Zhang: {yz['volatility_%']:.2f}% (Ann: {yz['annualized_%']:.1f}%)")

        realized = vol_predictor.calculate_realized_volatility()
        print(f"Realized Volatility: {realized['volatility_%']:.2f}% (Ann: {realized['annualized_%']:.1f}%)")

        print("\n--- Volatility Ratios ---")
        ratios = vol_predictor.calculate_volatility_ratios()
        print(f"10d/60d Ratio: {ratios['short_to_long_ratio']:.2f} ({ratios['interpretation']['short_to_long']})")
        print(f"10d/20d Ratio: {ratios['vol_10d_to_20d']:.2f}")
        print(f"Parkinson/Close Ratio: {ratios['parkinson_to_close']:.2f}")

        print("\n--- Volatility Skew ---")
        skew = vol_predictor.calculate_volatility_skew()
        print(f"Up Days Vol: {skew['volatility_up_days_%']:.2f}%")
        print(f"Down Days Vol: {skew['volatility_down_days_%']:.2f}%")
        print(f"Skew Ratio: {skew['skew_ratio']:.2f} - {skew['interpretation']}")
        print(f"Meaning: {skew['meaning']}")

        print("\n--- Volatility Cones ---")
        cones = vol_predictor.calculate_volatility_cones()
        if not cones.empty:
            print(cones[['Window', 'Current_%', 'Median_%', 'Percentile']].to_string(index=False))

        print("\n--- Volatility Term Structure ---")
        term_struct = vol_predictor.calculate_volatility_term_structure()
        if not term_struct.empty:
            print(term_struct.to_string(index=False))
            if 'shape' in term_struct.attrs:
                print(f"Term Structure Shape: {term_struct.attrs['shape']}")

        print("\n--- Volatility Regime ---")
        regime = vol_predictor.detect_volatility_regime()
        print(f"Current Volatility: {regime['current_volatility']:.2f}%")
        print(f"Regime: {regime['regime']}")
        print(f"Trend: {regime['trend']}")
        print(f"Percentile: {regime['percentile']:.1f}%")
        print(f"Interpretation: {regime['interpretation']}")

        print("\n--- GARCH Volatility Forecast ---")
        garch = vol_predictor.predict_volatility_garch(horizon=5)
        print(f"Method: {garch['method']}")
        print(f"Current Volatility: {garch['current_volatility']:.2f}%")
        print(f"Mean Forecast: {garch['mean_forecast']:.2f}%")
        print(f"5-Period Forecasts: {[f'{v:.2f}%' for v in garch['forecasts']]}")

        print("\n--- EWMA Volatility Forecast ---")
        ewma = vol_predictor.predict_volatility_ewma(horizon=5)
        print(f"Method: {ewma['method']}")
        print(f"Current Volatility: {ewma['current_volatility']:.2f}%")
        print(f"Forecast: {ewma['mean_forecast']:.2f}%")

        print("\n--- Volatility Clustering ---")
        clustering = vol_predictor.detect_volatility_clustering()
        print(f"Clustering Strength: {clustering['clustering_strength']}")
        print(f"Interpretation: {clustering['interpretation']}")
        print(f"Average Autocorrelation: {clustering['average_autocorr']:.3f}")

        print("\n--- Breakout Probability ---")
        breakout = vol_predictor.calculate_breakout_probability()
        print(f"Breakout Probability: {breakout['breakout_probability_%']:.1f}%")
        print(f"Direction Bias: {breakout['direction_bias']}")
        print(f"Position in Range: {breakout['position_in_range_%']:.1f}%")
        print(f"Volatility Status: {breakout['volatility_status']}")
        print(f"Squeeze Indicator: {breakout['squeeze_indicator']}")

        print("\n--- Risk Metrics ---")
        var_95 = vol_predictor.calculate_var(confidence_level=0.95, horizon=1)
        print(f"VaR (95% confidence):")
        print(f"  Historical: {var_95['historical_var_%']:.2f}% (${abs(var_95['historical_var_$']):.2f})")
        print(f"  Parametric: {var_95['parametric_var_%']:.2f}% (${abs(var_95['parametric_var_$']):.2f})")

        es = vol_predictor.calculate_expected_shortfall(confidence_level=0.95)
        print(f"\nExpected Shortfall (95%):")
        print(f"  {es['expected_shortfall_%']:.2f}% (${abs(es['expected_shortfall_$']):.2f})")
        print(f"  {es['interpretation']}")

        results['volatility_predictions'] = {
            'measurements': vol_predictor.get_all_volatility_measurements(),
            'regime': regime,
            'garch_forecast': garch,
            'ewma_forecast': ewma,
            'clustering': clustering,
            'breakout': breakout,
            'var_95': var_95,
            'expected_shortfall': es
        }

        print("\n✓ Volatility predictions completed successfully")

    except Exception as e:
        print(f"\n✗ Error in volatility predictions: {e}")
        results['volatility_predictions'] = {'error': str(e)}

    # ============================================================================
    # PATTERN RECOGNITION
    # ============================================================================
    print_section("4. PATTERN RECOGNITION")

    try:
        recognizer = PatternRecognizer(data_path)

        print("\n--- Similar Historical Patterns ---")
        similar = recognizer.find_similar_patterns(window=20, top_n=5)
        print(f"Bullish Ratio: {similar['bullish_ratio']:.1%}")
        print("Average Expected Returns:")
        for period, ret in similar['average_expected_returns'].items():
            print(f"  {period}: {ret:+.2f}%")

        print("\n--- Candlestick Patterns ---")
        candles = recognizer.detect_candlestick_patterns()
        print(f"Patterns Found: {candles['count']}")
        print(f"Overall Signal: {candles['overall_signal']}")
        if candles['patterns_found']:
            print("Detected:")
            for pattern in candles['patterns_found'][:3]:
                print(f"  • {pattern['pattern']} ({pattern['type']}) - {pattern['reliability']}%")

        print("\n--- Chart Patterns ---")
        charts = recognizer.detect_chart_patterns()
        print(f"Patterns Found: {charts['count']}")
        if charts['patterns_found']:
            for pattern in charts['patterns_found']:
                print(f"  • {pattern['pattern']} - {pattern['type']}")

        print("\n--- Support & Resistance ---")
        sr = recognizer.identify_support_resistance()
        print(f"Current Price: ${sr['current_price']:.2f}")
        print(f"Position in Range: {sr['position_in_range_%']:.1f}%")

        if sr['resistance_levels']:
            print("\nKey Resistance:")
            for level in sr['resistance_levels'][:2]:
                print(f"  ${level['level']:.2f} (+{level['distance_%']:.2f}%)")

        if sr['support_levels']:
            print("\nKey Support:")
            for level in sr['support_levels'][:2]:
                print(f"  ${level['level']:.2f} (-{level['distance_%']:.2f}%)")

        results['pattern_recognition'] = {
            'similar_patterns': similar,
            'candlestick': candles,
            'chart_patterns': charts,
            'support_resistance': sr
        }

        print("\n✓ Pattern recognition completed successfully")

    except Exception as e:
        print(f"\n✗ Error in pattern recognition: {e}")
        results['pattern_recognition'] = {'error': str(e)}

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print_section("ANALYSIS COMPLETE")
    print("\nAll analyses have been completed.")

    # Export to JSON if requested
    if export_json:
        output_file = Path(data_path).stem + "_analysis.json"
        output_path = Path(__file__).parent / output_file

        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                               np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results_json = convert_types(results)

        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)

        print(f"\n✓ Results exported to: {output_path}")

    return results


if __name__ == "__main__":
    # Check if file path provided as argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        data_file = sys.argv[1]
        print(f"\nUsing file from command line: {data_file}")
    else:
        # Interactive file selection
        data_file = select_data_file()

    # Check if export flag is set
    export = "--export" in sys.argv or "-e" in sys.argv

    try:
        run_complete_analysis(data_file, export_json=export)
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_file}")
        print("\nUsage: python orchestrator.py [path_to_csv] [--export]")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)