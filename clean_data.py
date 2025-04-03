import re
import pandas as pd
import nltk
import spacy
from pymongo import MongoClient
from nltk.corpus import stopwords

# Download required resources
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["insightlens"]
raw_col = db["raw_reviews"]
clean_col = db["clean_reviews"]

# Load reviews from MongoDB
print("🔄 Loading reviews from MongoDB...")
docs = list(raw_col.find({}, {"_id": 0, "Text": 1, "Score": 1}))
df = pd.DataFrame(docs)
print(f"✅ Loaded {len(df)} reviews")

# Clean review text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

df["clean_text"] = df["Text"].apply(clean_text)

# Label sentiment
def label_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df["sentiment"] = df["Score"].apply(label_sentiment)

# Save cleaned results into a new MongoDB collection
print("💾 Saving cleaned data to MongoDB...")
clean_col.delete_many({})
clean_col.insert_many(df.to_dict(orient="records"))
print("✅ Cleaned data saved to 'clean_reviews' collection.")