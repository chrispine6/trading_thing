from arango import ArangoClient
from typing import Optional
import os

class ArangoDBManager:
    # manages arangodb connection and operations for fmp data
    def __init__(
            self,
            host: str = "https://localhost:8529",
            username: str = "root",
            password: str = "mattdamon",
            db_name: str = "stark_trading_thing",
    ):
        """
        initialize ArangoDB connection
        :param host: arangodb server url
        :param username: db username
        :param password: db pwd
        :param db_name: name of db to connect to
        """
        self.client = ArangoClient(host=host)
        self.username=username
        self.password=password
        self.db_name=db_name
        self.db = None
        self.collection=None

    def connect(self):
        # establish connection and set up db/collection
        try:
            sys_db = self.client.db('_system', username=self.username, password=self.password)
            # create if does not exist
            if not sys_db.has_database(self.db_name):
                sys_db.create_database(self.db_name)
                print(f"created database: {self.db_name}")
            # connect to target
            self.db = self.client.db(self.db_name, username=self.username, password=self.password)
            if not self.db.has_collection('company_profiles'):
                self.collection = self.db.create_collection('company_profiles')
                print("created collection: company_profiles")

                # create indexes for efficient querying
                self.collection.add_hash_index(fields=['symbol'], unique=True)
                self.collection.add_hash_index(fields=['exchange'])
                self.collection.add_hash_index(fields=['sector'])
                self.collection.add_hash_index(fields=['industry'])
                print("created indexes on symbol, exchange, sector, and industry")
            else:
                self.collection = self.db.collection('company_profiles')
            print("successfully connected to arango")
            return True

        except Exception as e:
            print(f"error connecting to arango: {e}")
            return False


