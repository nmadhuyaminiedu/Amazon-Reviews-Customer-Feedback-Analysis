import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["insightlens"]
collection = db["clean_reviews"]

# Load data
print("🔄 Loading data from MongoDB...")
docs = list(collection.find({}, {"_id": 0}))  # include all fields
df = pd.DataFrame(docs)

# Save to CSV
csv_path = "C:\\Users\\nmadh\\OneDrive\\Desktop\\NLP\\clean_reviews.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Exported cleaned data to {csv_path}")