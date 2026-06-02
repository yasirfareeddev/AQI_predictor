import os
import time
from pymongo import MongoClient, errors
from dotenv import load_dotenv

load_dotenv()

def get_mongo_client():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

    for attempt in range(3):
        try:
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
                retryReads=True
            )

            client.admin.command("ping")
            print("MongoDB connected")

            return client["aqi_db"]

        except errors.ServerSelectionTimeoutError as e:
            print(f"MongoDB connection failed (attempt {attempt+1}/3)")

            if attempt < 2:
                time.sleep(5)
            else:
                raise e