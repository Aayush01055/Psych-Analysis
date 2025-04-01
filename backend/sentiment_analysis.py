import pickle
import os

model_path = os.path.join("models", "sentiment_model.pkl")

# Load trained sentiment model
with open(model_path, "rb") as file:
    sentiment_model = pickle.load(file)

def predict_sentiment(text):
    return sentiment_model.predict([text])[0]
