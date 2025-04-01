import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/emotion_dataset.csv")  # Replace with actual dataset
df = df.dropna()
df = df[df['Emotion'].isin(['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust'])]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["Response"], df["Emotion"], test_size=0.2, random_state=42)

# Create pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Emotion Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
with open("models/emotion_model.pkl", "wb") as file:
    pickle.dump(model, file)
