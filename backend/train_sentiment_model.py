import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/sentiment_dataset.csv")  # Replace with actual dataset
df = df.dropna()
df = df[df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["Response"], df["Sentiment"], test_size=0.2, random_state=42)

# Create pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Sentiment Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
with open("models/sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)
