import requests
from typing import List, Optional
from arango_connection import ArangoDBManager


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
        self.base_url = "https://financialmodelingprep.com/api/v3"
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
            url = f"{self.base_url}/profile/{symbol}"
            params = {"apikey": self.api_key}

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data and len(data) > 0:
                return data[0]  # API returns list with single item
            else:
                print(f"No data found for symbol: {symbol}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching profile for {symbol}: {e}")
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

    def fetch_and_save_multiple_profiles(self, symbols: List[str]) -> dict:
        """
        Fetch multiple company profiles and save to ArangoDB

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary with success/failure counts
        """
        results = {
            'success': 0,
            'failed': 0,
            'failed_symbols': []
        }

        for symbol in symbols:
            if self.fetch_and_save_profile(symbol):
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_symbols'].append(symbol)

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

    def update_all_exchange_profiles(self, exchange: str = "NSE", batch_size: int = 50) -> dict:
        """
        Fetch and update all company profiles for an exchange

        Args:
            exchange: Exchange code
            batch_size: Number of symbols to process in each batch

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
            print(f"\nProcessing batch {i // batch_size + 1}: symbols {i + 1}-{min(i + batch_size, len(symbols))}")

            batch_results = self.fetch_and_save_multiple_profiles(batch)
            total_results['success'] += batch_results['success']
            total_results['failed'] += batch_results['failed']
            total_results['failed_symbols'].extend(batch_results['failed_symbols'])

        return total_results


# Example usage
if __name__ == "__main__":
    # Setup ArangoDB connection
    db_manager = ArangoDBManager(
        host="http://localhost:8529",
        username="root",
        password="openSesame",
        db_name="fmp_data"
    )

    if not db_manager.connect():
        print("Failed to connect to ArangoDB")
        exit(1)

    # Initialize FMP fetcher
    FMP_API_KEY = "your_api_key_here"  # Replace with your actual API key
    fetcher = FMPDataFetcher(FMP_API_KEY, db_manager)

    # Example 1: Fetch single company
    print("Fetching single company profile...")
    fetcher.fetch_and_save_profile("BEL.NS")

    # Example 2: Fetch multiple companies
    print("\nFetching multiple company profiles...")
    indian_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    results = fetcher.fetch_and_save_multiple_profiles(indian_stocks)
    print(f"Results: {results}")

    # Example 3: Retrieve saved data
    print("\nRetrieving saved profile...")
    profile = db_manager.get_company_by_symbol("BEL.NS")
    if profile:
        print(f"Company: {profile.get('companyName')}")
        print(f"Sector: {profile.get('sector')}")
        print(f"Market Cap: {profile.get('marketCap')}")

    # Example 4: Query by sector
    print("\nQuerying companies in Industrials sector...")
    industrials = db_manager.get_companies_by_sector("Industrials")
    print(f"Found {len(industrials)} companies in Industrials sector")

    # Close connection
    db_manager.close()