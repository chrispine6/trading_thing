from arango import ArangoClient
from typing import Optional, List, Dict, Literal
import certifi
import json
import ssl
from urllib.request import urlopen
from datetime import datetime


class FundamentalsDB:
    """
    Manages financial statements data in ArangoDB with FMP API integration
    Collections: income_statements, balance_sheets, cash_flow_statements
    """

    def __init__(
            self,
            host: str = "http://localhost:8529",
            username: str = "root",
            password: str = "mattdamon",
            db_name: str = "fmp_data",
            api_key: str = None
    ):
        """
        Initialize ArangoDB connection and FMP API key

        Args:
            host: ArangoDB server URL
            username: Database username
            password: Database password
            db_name: Database name
            api_key: Financial Modeling Prep API key
        """
        self.client = ArangoClient(hosts=host)
        self.username = username
        self.password = password
        self.db_name = db_name
        self.api_key = api_key
        self.db = None

        # Collection references
        self.income_statements = None
        self.balance_sheets = None
        self.cash_flow_statements = None

        # FMP API endpoints
        self.api_endpoints = {
            'income': 'https://financialmodelingprep.com/stable/income-statement',
            'balance': 'https://financialmodelingprep.com/stable/balance-sheet-statement',
            'cashflow': 'https://financialmodelingprep.com/stable/cash-flow-statement'
        }

    def connect(self):
        """Establish connection and setup database/collections"""
        try:
            # Connect to system database
            sys_db = self.client.db('_system', username=self.username, password=self.password)

            # Create database if it doesn't exist
            if not sys_db.has_database(self.db_name):
                sys_db.create_database(self.db_name)
                print(f"Created database: {self.db_name}")

            # Connect to the target database
            self.db = self.client.db(self.db_name, username=self.username, password=self.password)

            # Setup collections
            self._setup_collection('income_statements')
            self._setup_collection('balance_sheets')
            self._setup_collection('cash_flow_statements')

            print("Successfully connected to ArangoDB")
            return True

        except Exception as e:
            print(f"Error connecting to ArangoDB: {e}")
            return False

    def _setup_collection(self, collection_name: str):
        """Create collection with appropriate indexes"""
        try:
            if not self.db.has_collection(collection_name):
                collection = self.db.create_collection(collection_name)
                print(f"Created collection: {collection_name}")

                # Create indexes for efficient querying
                collection.add_hash_index(fields=['symbol'], unique=False)
                collection.add_hash_index(fields=['date'], unique=False)
                collection.add_hash_index(fields=['symbol', 'date'], unique=True)
                print(f"Created indexes on {collection_name}")
            else:
                collection = self.db.collection(collection_name)

            # Store collection reference
            if collection_name == 'income_statements':
                self.income_statements = collection
            elif collection_name == 'balance_sheets':
                self.balance_sheets = collection
            elif collection_name == 'cash_flow_statements':
                self.cash_flow_statements = collection

        except Exception as e:
            print(f"Error setting up collection {collection_name}: {e}")

    def _get_jsonparsed_data(self, url: str) -> dict:
        """Fetch and parse JSON data from URL"""
        context = ssl.create_default_context(cafile=certifi.where())
        response = urlopen(url, context=context)
        data = response.read().decode("utf-8")
        return json.loads(data)

    def _fetch_from_fmp(self, symbol: str, statement_type: Literal['income', 'balance', 'cashflow']) -> List[dict]:
        """
        Fetch financial statements from FMP API

        Args:
            symbol: Company ticker symbol
            statement_type: Type of statement to fetch

        Returns:
            List of financial statement periods
        """
        if not self.api_key:
            raise ValueError("API key not provided")

        url = f"{self.api_endpoints[statement_type]}?symbol={symbol}&apikey={self.api_key}"

        try:
            data = self._get_jsonparsed_data(url)
            print(f"✓ Fetched {len(data)} periods of {statement_type} data for {symbol}")
            return data
        except Exception as e:
            print(f"✗ Error fetching {statement_type} data for {symbol}: {e}")
            return []

    def _get_collection_for_type(self, statement_type: str):
        """Get the appropriate collection for statement type"""
        mapping = {
            'income': self.income_statements,
            'balance': self.balance_sheets,
            'cashflow': self.cash_flow_statements
        }
        return mapping.get(statement_type)

    def _store_statements(self, symbol: str, statements: List[dict], statement_type: str) -> int:
        """
        Store financial statements in database

        Args:
            symbol: Company ticker symbol
            statements: List of statement periods from FMP
            statement_type: Type of statement

        Returns:
            Number of successfully stored documents
        """
        collection = self._get_collection_for_type(statement_type)
        if not collection:
            print(f"Error: Unknown statement type {statement_type}")
            return 0

        success_count = 0
        for statement in statements:
            try:
                # Create document key from symbol and date
                date = statement.get('date', '')
                doc_key = f"{symbol.replace('.', '_').replace('-', '_')}_{date.replace('-', '_')}"

                document = {
                    '_key': doc_key,
                    'symbol': symbol,
                    **statement
                }

                # Insert or update
                collection.insert(document, overwrite=True, return_new=True)
                success_count += 1

            except Exception as e:
                print(f"✗ Error storing {statement_type} for {symbol} on {statement.get('date')}: {e}")

        print(f"✓ Stored {success_count}/{len(statements)} {statement_type} records for {symbol}")
        return success_count

    def _get_latest_date(self, symbol: str, statement_type: str) -> Optional[str]:
        """Get the most recent date available in database for a symbol"""
        collection = self._get_collection_for_type(statement_type)
        if not collection:
            return None

        try:
            query = """
                FOR doc IN @@collection
                    FILTER doc.symbol == @symbol
                    SORT doc.date DESC
                    LIMIT 1
                    RETURN doc.date
            """
            cursor = self.db.aql.execute(
                query,
                bind_vars={'@collection': collection.name, 'symbol': symbol}
            )
            result = list(cursor)
            return result[0] if result else None

        except Exception as e:
            print(f"Error getting latest date: {e}")
            return None

    def get_financials(
            self,
            symbol: str,
            statement_type: Literal['income', 'balance', 'cashflow', 'all'] = 'all',
            mode: Literal['latest', 'all', 'range'] = 'latest',
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            force_refresh: bool = False
    ) -> Dict[str, List[dict]]:
        """
        Get financial statements for a company

        Args:
            symbol: Company ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cashflow', 'all')
            mode: 'latest' (most recent), 'all' (all historical), 'range' (date range)
            start_date: Start date for range mode (YYYY-MM-DD)
            end_date: End date for range mode (YYYY-MM-DD)
            force_refresh: Force fetch from API even if data exists

        Returns:
            Dictionary with statement types as keys and lists of statements as values
        """
        statement_types = ['income', 'balance', 'cashflow'] if statement_type == 'all' else [statement_type]
        result = {}

        for stmt_type in statement_types:
            # Check if we need to fetch from API
            latest_date = self._get_latest_date(symbol, stmt_type)

            should_fetch = force_refresh or latest_date is None

            # Fetch from API if needed
            if should_fetch:
                print(f"Fetching {stmt_type} data from FMP API for {symbol}...")
                statements = self._fetch_from_fmp(symbol, stmt_type)
                if statements:
                    self._store_statements(symbol, statements, stmt_type)

            # Retrieve from database based on mode
            result[stmt_type] = self._get_statements_from_db(symbol, stmt_type, mode, start_date, end_date)

        return result

    def _get_statements_from_db(
            self,
            symbol: str,
            statement_type: str,
            mode: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> List[dict]:
        """Retrieve statements from database based on mode"""
        collection = self._get_collection_for_type(statement_type)
        if not collection:
            return []

        try:
            if mode == 'latest':
                query = """
                    FOR doc IN @@collection
                        FILTER doc.symbol == @symbol
                        SORT doc.date DESC
                        LIMIT 1
                        RETURN doc
                """
                bind_vars = {'@collection': collection.name, 'symbol': symbol}

            elif mode == 'all':
                query = """
                    FOR doc IN @@collection
                        FILTER doc.symbol == @symbol
                        SORT doc.date DESC
                        RETURN doc
                """
                bind_vars = {'@collection': collection.name, 'symbol': symbol}

            elif mode == 'range':
                if not start_date or not end_date:
                    raise ValueError("start_date and end_date required for range mode")

                query = """
                    FOR doc IN @@collection
                        FILTER doc.symbol == @symbol
                        FILTER doc.date >= @start_date
                        FILTER doc.date <= @end_date
                        SORT doc.date DESC
                        RETURN doc
                """
                bind_vars = {
                    '@collection': collection.name,
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                }

            cursor = self.db.aql.execute(query, bind_vars=bind_vars)
            return list(cursor)

        except Exception as e:
            print(f"Error retrieving {statement_type} statements: {e}")
            return []

    def close(self):
        """Close the database connection"""
        self.client.close()
        print("Connection closed")


# Example usage
if __name__ == "__main__":
    # Initialize with your API key
    db = FundamentalsDB(api_key="YOUR_API_KEY")

    if db.connect():
        # Get latest financials (fetches from API if not in DB)
        financials = db.get_financials('AAPL', statement_type='all', mode='latest')
        print(f"\nLatest Income Statement: {financials['income'][0]['date'] if financials['income'] else 'None'}")

        # Get all historical data
        all_data = db.get_financials('AAPL', statement_type='income', mode='all')
        print(f"\nTotal historical income statements: {len(all_data['income'])}")

        # Get specific date range
        range_data = db.get_financials(
            'AAPL',
            statement_type='balance',
            mode='range',
            start_date='2023-01-01',
            end_date='2024-12-31'
        )
        print(f"\nBalance sheets in range: {len(range_data['balance'])}")

        db.close()