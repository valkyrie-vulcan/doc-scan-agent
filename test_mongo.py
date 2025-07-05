# test_mongo.py
import os
import pymongo
from dotenv import load_dotenv

print("--- Starting MongoDB Connection Test ---")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("FATAL: MONGO_URI not found in .env file. Please check your configuration.")
else:
    print(f"Attempting to connect to: {MONGO_URI}")
    try:
        # We use a short timeout to fail faster if there's a problem.
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        
        print("\n✅ SUCCESS: MongoDB connection established successfully!")

    except pymongo.errors.ConnectionFailure as e:
        print(f"\n❌ FAILED: Could not connect to MongoDB.")
        print(f"   Error Details: {e}")
        print("\n   This is likely a FIREWALL issue. Please follow the steps to create a firewall rule.")
    except Exception as e:
        print(f"\n❌ FAILED: An unexpected error occurred.")
        print(f"   Error Details: {e}")

print("\n--- Test Complete ---")
