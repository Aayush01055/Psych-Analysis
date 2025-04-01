import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("/workspaces/Psych-Analysis/backend/data/sentiment_emotion_psych_dataset.csv")  # Ensure correct path
df = df.dropna()

# Ensure correct column names exist
if 'Sentiment' not in df.columns or 'Response' not in df.columns:
    raise ValueError("Dataset must contain 'Sentiment' and 'Response' columns.")

# Filter for valid sentiment labels
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
with open("/workspaces/Psych-Analysis/backend/models/sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Sentiment model trained and saved successfully!")
