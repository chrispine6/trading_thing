"""
Orchestrator for fundamentals retrieval and storage system

This is the main file to execute. It fetches company profiles from FMP API
and stores them in ArangoDB.

Usage:
    python orchestrator.py
"""

from db_connector import ArangoDBManager
from company_profile_db import FMPDataFetcher
from datetime import datetime


# ===========================
# CONFIGURATION
# ===========================

# Database Configuration
DB_CONFIG = {
    "host": "http://localhost:8529",
    "username": "root",
    "password": "mattdamon",  # Change if you modified your ArangoDB password
    "db_name": "fmp_data"
}

# API Configuration
FMP_API_KEY = "yx7LVs66th1fvnvSAqsMGUCXICOc79ch"

# Symbols to fetch (Add/modify as needed)
SYMBOLS_TO_FETCH = [
    # Indian Stocks (NSE)
    "BEL.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "SBIN.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "BAJFINANCE.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "ULTRACEMCO.NS",
    "TITAN.NS",

    # US Stocks (You can mix exchanges)
    # "AAPL",
    # "MSFT",
    # "GOOGL",
    # "AMZN",
    # "TSLA",
]

# Processing Configuration
BATCH_SIZE = 20  # Number of symbols to process at once
API_DELAY = 0.3  # Delay between API calls in seconds (to avoid rate limits)


# ===========================
# MAIN EXECUTION FUNCTIONS
# ===========================

def initialize_system():
    """Initialize database connection and data fetcher"""
    print("=" * 70)
    print("COMPANY PROFILE FETCHER - INITIALIZATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup ArangoDB connection
    print("Connecting to ArangoDB...")
    db_manager = ArangoDBManager(**DB_CONFIG)

    if not db_manager.connect():
        print("❌ Failed to connect to ArangoDB")
        print("\nTroubleshooting:")
        print("1. Ensure ArangoDB is running (docker ps)")
        print("2. Check the connection details in DB_CONFIG")
        print("3. Verify the password is correct")
        return None, None

    print("✓ ArangoDB connected successfully\n")

    # Initialize FMP fetcher
    print("Initializing FMP Data Fetcher...")
    fetcher = FMPDataFetcher(FMP_API_KEY, db_manager)
    print("✓ FMP Data Fetcher initialized\n")

    return db_manager, fetcher


def fetch_specified_symbols(fetcher, symbols):
    """Fetch and store profiles for specified symbols"""
    print("=" * 70)
    print(f"FETCHING {len(symbols)} COMPANY PROFILES")
    print("=" * 70)
    print(f"Symbols to process: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}\n")

    results = fetcher.fetch_and_save_multiple_profiles(symbols, delay=API_DELAY)

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Successfully saved: {results['success']}")
    print(f"✗ Failed: {results['failed']}")

    if results['failed_symbols']:
        print(f"\nFailed symbols: {', '.join(results['failed_symbols'])}")

    return results


def fetch_entire_exchange(fetcher, exchange="NSE"):
    """Fetch all companies from a specific exchange"""
    print("=" * 70)
    print(f"FETCHING ALL COMPANIES FROM {exchange} EXCHANGE")
    print("=" * 70)
    print("⚠ Warning: This may take a long time and use many API calls\n")

    results = fetcher.update_all_exchange_profiles(
        exchange=exchange,
        batch_size=BATCH_SIZE,
        delay=API_DELAY
    )

    print("\n" + "=" * 70)
    print("EXCHANGE UPDATE COMPLETE")
    print("=" * 70)
    print(f"✓ Successfully saved: {results['success']}")
    print(f"✗ Failed: {results['failed']}")

    return results


def display_sample_data(db_manager, num_samples=3):
    """Display sample data from the database"""
    print("\n" + "=" * 70)
    print("SAMPLE DATA FROM DATABASE")
    print("=" * 70)

    companies = db_manager.get_all_companies()

    if not companies:
        print("No companies found in database.")
        return

    print(f"Total companies in database: {len(companies)}\n")

    for i, company in enumerate(companies[:num_samples], 1):
        print(f"{i}. {company.get('companyName', 'N/A')} ({company.get('symbol', 'N/A')})")
        print(f"   Sector: {company.get('sector', 'N/A')}")
        print(f"   Industry: {company.get('industry', 'N/A')}")
        print(f"   Exchange: {company.get('exchange', 'N/A')}")
        print(f"   Market Cap: {company.get('marketCap', 'N/A'):,}" if company.get('marketCap') else "   Market Cap: N/A")
        print()


def query_by_sector(db_manager, sector):
    """Query and display companies by sector"""
    print(f"\nQuerying companies in {sector} sector...")
    companies = db_manager.get_companies_by_sector(sector)
    print(f"Found {len(companies)} companies in {sector} sector")

    for company in companies[:5]:  # Show first 5
        print(f"  - {company.get('companyName')} ({company.get('symbol')})")


# ===========================
# MAIN EXECUTION
# ===========================

def main():
    """Main execution function"""
    # Initialize system
    db_manager, fetcher = initialize_system()

    if not db_manager or not fetcher:
        return

    try:
        # Option 1: Fetch specific symbols (DEFAULT)
        results = fetch_specified_symbols(fetcher, SYMBOLS_TO_FETCH)

        # Option 2: Fetch entire exchange (UNCOMMENT TO USE)
        # Note: This will fetch ALL companies from the exchange
        # results = fetch_entire_exchange(fetcher, exchange="NSE")

        # Display sample data
        if results['success'] > 0:
            display_sample_data(db_manager, num_samples=5)

            # Example: Query by sector (if you have companies in that sector)
            # query_by_sector(db_manager, "Financial Services")

    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        # Close database connection
        print("\n" + "=" * 70)
        print("SHUTTING DOWN")
        print("=" * 70)
        db_manager.close()
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()