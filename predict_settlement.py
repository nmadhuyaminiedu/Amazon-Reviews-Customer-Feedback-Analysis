import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("✅ Model, vectorizer, and encoder loaded successfully!")

# Prediction loop
while True:
    review = input("\n📝 Enter a review (or type 'exit' to quit):\n> ")

    if review.lower() == "exit":
        print("👋 Exiting sentiment checker.")
        break

    # Vectorize and predict
    X_input = vectorizer.transform([review])
    y_pred = model.predict(X_input)
    sentiment = label_encoder.inverse_transform(y_pred)

    print(f"🔍 Predicted Sentiment: **{sentiment[0]}**")
