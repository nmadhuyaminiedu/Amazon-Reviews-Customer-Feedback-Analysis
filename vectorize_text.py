from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from load_data import load_cleaned_reviews

# Step 1: Load cleaned data
df = load_cleaned_reviews()

# Step 2: TF-IDF Vectorization
print("⚙️ Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# Step 3: Encode Sentiment Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

# Output
print("✅ TF-IDF Matrix Shape:", X.shape)
print("✅ Encoded Labels Sample:", y[:10])
print("✅ Sentiment Classes:", label_encoder.classes_)