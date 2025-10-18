import requests
from typing import List, Optional
from db_connector import ArangoDBManager
import time

"""
    fmp data fetcher
        1. fetch company profile
        2. fetch and save profile
        3. fetch and save multiple profiles
        4. fetch exchange profiles
        5. update all exchange profiles
"""

class FMPDataFetcher:
    """Fetches data from Financial Modeling Prep API and stores in ArangoDB"""

    def __init__(self, api_key: str, db_manager: ArangoDBManager):
        """
        Initialize FMP data fetcher

        Args:
            api_key: Your FMP API key
            db_manager: ArangoDBManager instance
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        self.db_manager = db_manager

    def fetch_company_profile(self, symbol: str) -> Optional[dict]:
        """
        Fetch company profile for a single symbol

        Args:
            symbol: Stock symbol (e.g., 'BEL.NS', 'AAPL')

        Returns:
            Company profile dictionary or None
        """
        try:
            url = f"{self.base_url}/profile"
            params = {
                "symbol": symbol,
                "apikey": self.api_key
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data and len(data) > 0:
                return data[0]  # API returns list with single item
            else:
                print(f"⚠ No data found for symbol: {symbol}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching profile for {symbol}: {e}")
            return None

    def fetch_and_save_profile(self, symbol: str) -> bool:
        """
        Fetch company profile and save to ArangoDB

        Args:
            symbol: Stock symbol

        Returns:
            True if successful, False otherwise
        """
        profile = self.fetch_company_profile(symbol)

        if profile:
            result = self.db_manager.insert_company_profile(profile)
            return result is not None

        return False

    def fetch_and_save_multiple_profiles(self, symbols: List[str], delay: float = 0.2) -> dict:
        """
        Fetch multiple company profiles and save to ArangoDB

        Args:
            symbols: List of stock symbols
            delay: Delay between API calls in seconds (to avoid rate limits)

        Returns:
            Dictionary with success/failure counts
        """
        results = {
            'success': 0,
            'failed': 0,
            'failed_symbols': []
        }

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Processing {symbol}...")

            if self.fetch_and_save_profile(symbol):
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_symbols'].append(symbol)

            # Add delay to avoid hitting API rate limits
            if i < len(symbols):
                time.sleep(delay)

        return results

    def fetch_exchange_symbols(self, exchange: str = "NSE") -> List[str]:
        """
        Fetch all available symbols from a specific exchange

        Args:
            exchange: Exchange code (e.g., 'NSE', 'NASDAQ')

        Returns:
            List of symbols
        """
        try:
            url = f"{self.base_url}/symbol/{exchange}"
            params = {"apikey": self.api_key}

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return [item['symbol'] for item in data if 'symbol' in item]

        except requests.exceptions.RequestException as e:
            print(f"Error fetching exchange symbols: {e}")
            return []

    def update_all_exchange_profiles(self, exchange: str = "NSE", batch_size: int = 50, delay: float = 0.2) -> dict:
        """
        Fetch and update all company profiles for an exchange

        Args:
            exchange: Exchange code
            batch_size: Number of symbols to process in each batch
            delay: Delay between API calls in seconds

        Returns:
            Dictionary with processing statistics
        """
        symbols = self.fetch_exchange_symbols(exchange)

        if not symbols:
            print(f"No symbols found for exchange: {exchange}")
            return {'success': 0, 'failed': 0}

        print(f"Found {len(symbols)} symbols for {exchange}")
        print(f"Processing in batches of {batch_size}...")

        total_results = {'success': 0, 'failed': 0, 'failed_symbols': []}

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            print(f"\n{'=' * 60}")
            print(f"Processing batch {i // batch_size + 1}: symbols {i + 1}-{min(i + batch_size, len(symbols))}")
            print(f"{'=' * 60}")

            batch_results = self.fetch_and_save_multiple_profiles(batch, delay=delay)
            total_results['success'] += batch_results['success']
            total_results['failed'] += batch_results['failed']
            total_results['failed_symbols'].extend(batch_results['failed_symbols'])

            print(f"\nBatch summary: {batch_results['success']} succeeded, {batch_results['failed']} failed")

        return total_results