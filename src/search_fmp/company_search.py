#!/usr/bin/env python
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
    import ssl
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib3 import urlopen

import certifi
import json

def get_jsonparsed_data(url):
    # Create SSL context with certifi certificates to avoid deprecation warning
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)

def display_results(results):
    """Display search results in a readable format"""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} result(s):\n")
    print("-" * 80)

    for idx, item in enumerate(results, 1):
        print(f"{idx}. {item['name']}")
        print(f"   Symbol: {item['symbol']}")
        print(f"   Exchange: {item['exchangeFullName']} ({item['exchange']})")
        print(f"   Currency: {item['currency']}")
        print("-" * 80)

# Interactive search
if __name__ == "__main__":
    api_key = "yx7LVs66th1fvnvSAqsMGUCXICOc79ch"

    search_query = input("Enter company symbol or name to search: ").strip()

    if not search_query:
        print("Error: Search query cannot be empty.")
    else:
        url = f"https://financialmodelingprep.com/stable/search-symbol?query={search_query}&apikey={api_key}"

        try:
            results = get_jsonparsed_data(url)
            display_results(results)
        except Exception as e:
            print(f"Error occurred: {e}")
