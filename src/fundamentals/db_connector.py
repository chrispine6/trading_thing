from arango import ArangoClient
from typing import Optional, List
import os


class ArangoDBManager:
    """Manages ArangoDB connection and operations for FMP data"""

    def __init__(
            self,
            host: str = "http://localhost:8529",
            username: str = "root",
            password: str = "mattdamon",
            db_name: str = "fmp_data"
    ):
        """
        Initialize ArangoDB connection

        Args:
            host: ArangoDB server URL (default Docker setup)
            username: Database username
            password: Database password
            db_name: Name of the database to use/create
        """
        self.client = ArangoClient(hosts=host)
        self.username = username
        self.password = password
        self.db_name = db_name
        self.db = None
        self.collection = None

    def connect(self):
        """Establish connection and setup database/collection"""
        try:
            # Connect to system database
            sys_db = self.client.db('_system', username=self.username, password=self.password)

            # Create database if it doesn't exist
            if not sys_db.has_database(self.db_name):
                sys_db.create_database(self.db_name)
                print(f"Created database: {self.db_name}")

            # Connect to the target database
            self.db = self.client.db(self.db_name, username=self.username, password=self.password)

            # Create collection if it doesn't exist
            if not self.db.has_collection('company_profiles'):
                self.collection = self.db.create_collection('company_profiles')
                print("Created collection: company_profiles")

                # Create indexes for efficient querying
                self.collection.add_hash_index(fields=['symbol'], unique=True)
                self.collection.add_hash_index(fields=['exchange'])
                self.collection.add_hash_index(fields=['sector'])
                self.collection.add_hash_index(fields=['industry'])
                print("Created indexes on symbol, exchange, sector, and industry")
            else:
                self.collection = self.db.collection('company_profiles')

            print("Successfully connected to ArangoDB")
            return True

        except Exception as e:
            print(f"Error connecting to ArangoDB: {e}")
            return False

    def insert_company_profile(self, profile_data: dict) -> Optional[str]:
        """
        Insert or update a company profile

        Args:
            profile_data: Dictionary containing company profile data from FMP

        Returns:
            Document key if successful, None otherwise
        """
        try:
            # Use symbol as the document key for easy updates
            symbol = profile_data.get('symbol')
            if not symbol:
                print("Error: No symbol in profile data")
                return None

            # ArangoDB document key requirements: replace dots and special chars
            doc_key = symbol.replace('.', '_').replace('-', '_')

            # Prepare document with _key
            document = {
                '_key': doc_key,
                **profile_data
            }

            # Insert or update (upsert operation)
            result = self.collection.insert(
                document,
                overwrite=True,  # Update if exists
                return_new=True
            )

            print(f"✓ Saved profile for {symbol}")
            return result['_key']

        except Exception as e:
            print(f"✗ Error inserting company profile for {profile_data.get('symbol', 'unknown')}: {e}")
            return None

    def insert_multiple_profiles(self, profiles: List[dict]) -> int:
        """
        Insert or update multiple company profiles

        Args:
            profiles: List of company profile dictionaries from FMP

        Returns:
            Number of successfully inserted documents
        """
        success_count = 0

        for profile in profiles:
            if self.insert_company_profile(profile):
                success_count += 1

        return success_count

    def get_company_by_symbol(self, symbol: str) -> Optional[dict]:
        """
        Retrieve company profile by symbol

        Args:
            symbol: Company symbol (e.g., 'BEL.NS')

        Returns:
            Company profile dictionary or None
        """
        try:
            doc_key = symbol.replace('.', '_').replace('-', '_')
            return self.collection.get(doc_key)
        except Exception as e:
            print(f"Error retrieving company {symbol}: {e}")
            return None

    def get_companies_by_sector(self, sector: str) -> List[dict]:
        """Get all companies in a specific sector"""
        try:
            query = """
                FOR doc IN company_profiles
                    FILTER doc.sector == @sector
                    RETURN doc
            """
            cursor = self.db.aql.execute(query, bind_vars={'sector': sector})
            return list(cursor)
        except Exception as e:
            print(f"Error querying by sector: {e}")
            return []

    def get_companies_by_industry(self, industry: str) -> List[dict]:
        """Get all companies in a specific industry"""
        try:
            query = """
                FOR doc IN company_profiles
                    FILTER doc.industry == @industry
                    RETURN doc
            """
            cursor = self.db.aql.execute(query, bind_vars={'industry': industry})
            return list(cursor)
        except Exception as e:
            print(f"Error querying by industry: {e}")
            return []

    def get_all_companies(self) -> List[dict]:
        """Get all companies in the database"""
        try:
            query = """
                FOR doc IN company_profiles
                    RETURN doc
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            print(f"Error retrieving all companies: {e}")
            return []

    def close(self):
        """Close the database connection"""
        self.client.close()
        print("Connection closed")