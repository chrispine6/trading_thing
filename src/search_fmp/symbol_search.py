#!/usr/bin/env python
try:
    from urllib.request import urlopen
    import ssl
except ImportError:
    from urllib3 import urlopen

import certifi
import json

def get_jsonparsed_data(url):
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)

def display_results(results):
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

if __name__ == "__main__":
    api_key = "yx7LVs66th1fvnvSAqsMGUCXICOc79ch"
    symbol_query = input("Enter symbol to search: ").strip()

    if not symbol_query:
        print("Error: Symbol query cannot be empty.")
    else:
        url = f"https://financialmodelingprep.com/stable/search-symbol?query={symbol_query}&apikey={api_key}"
        try:
            results = get_jsonparsed_data(url)
            display_results(results)
        except Exception as e:
            print(f"Error occurred: {e}")

