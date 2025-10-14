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

def display_profile(profile):
    if not profile:
        print("No profile found.")
        return

    item = profile[0]  # API returns a list with one dict
    print(f"\nCompany Profile for {item.get('symbol', 'N/A')}:")
    print("-" * 80)
    print(f"Name: {item.get('companyName', 'N/A')}")
    print(f"Industry: {item.get('industry', 'N/A')}")
    print(f"Sector: {item.get('sector', 'N/A')}")
    print(f"Exchange: {item.get('exchangeShortName', 'N/A')}")
    print(f"Website: {item.get('website', 'N/A')}")
    print(f"Description: {item.get('description', 'N/A')}")
    print(f"CEO: {item.get('ceo', 'N/A')}")
    print(f"Price: {item.get('price', 'N/A')}")
    print(f"Currency: {item.get('currency', 'N/A')}")
    print("-" * 80)

if __name__ == "__main__":
    api_key = "yx7LVs66th1fvnvSAqsMGUCXICOc79ch"
    symbol = input("Enter symbol to fetch company profile: ").strip().upper()

    if not symbol:
        print("Error: Symbol cannot be empty.")
    else:
        url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={api_key}"
        try:
            profile = get_jsonparsed_data(url)
            display_profile(profile)
        except Exception as e:
            print(f"Error occurred: {e}")

