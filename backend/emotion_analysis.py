import pickle
import os

model_path = os.path.join("models", "emotion_model.pkl")

# Load trained emotion model
with open(model_path, "rb") as file:
    emotion_model = pickle.load(file)

def predict_emotion(text):
    return emotion_model.predict([text])[0]
