from arango import ArangoClient

# Connect to ArangoDB
client = ArangoClient(hosts="http://localhost:8529")
db = client.db('fmp_data', username='root', password='mattdamon')

collections = ['income_statements', 'balance_sheets', 'cash_flow_statements']

for collection_name in collections:
    print(f"\nProcessing {collection_name}...")
    collection = db.collection(collection_name)

    # Get all indexes
    indexes = collection.indexes()

    print(f"Current indexes:")
    for idx in indexes:
        print(f"  - {idx['type']}: {idx.get('fields', [])} (unique: {idx.get('unique', False)})")

    # Delete the problematic index with ['symbol', 'date']
    for idx in indexes:
        fields = idx.get('fields', [])
        if 'date' in fields or (len(fields) == 2 and 'symbol' in fields and 'fiscalDateEnding' in fields):
            print(f"  Deleting index: {idx['id']}")
            try:
                collection.delete_index(idx['id'])
                print(f"  ✓ Deleted")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # Recreate correct indexes
    try:
        collection.add_hash_index(fields=['symbol'], unique=False)
        print(f"  ✓ Created index on ['symbol']")
    except Exception as e:
        print(f"  Index on ['symbol'] already exists or error: {e}")

    try:
        collection.add_hash_index(fields=['fiscalDateEnding'], unique=False)
        print(f"  ✓ Created index on ['fiscalDateEnding']")
    except Exception as e:
        print(f"  Index on ['fiscalDateEnding'] already exists or error: {e}")

    try:
        collection.add_hash_index(fields=['symbol', 'fiscalDateEnding', 'reportType'], unique=True)
        print(f"  ✓ Created unique index on ['symbol', 'fiscalDateEnding', 'reportType']")
    except Exception as e:
        print(f"  Unique index already exists or error: {e}")

print("\n✓ Index fix complete!")
client.close()