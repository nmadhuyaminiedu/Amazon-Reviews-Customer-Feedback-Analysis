import pandas as pd
from pymongo import MongoClient

def load_cleaned_reviews():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["insightlens"]
    collection = db["clean_reviews"]

    # Load clean_text and sentiment fields
    print("🔄 Loading cleaned reviews from MongoDB...")
    docs = list(collection.find({}, {"_id": 0, "clean_text": 1, "sentiment": 1}))
    df = pd.DataFrame(docs).dropna(subset=["clean_text", "sentiment"])
    print(f"✅ Loaded {len(df)} records from MongoDB")
    return df