import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "/workspaces/Psych-Analysis/backend/data/sentiment_emotion_psych_dataset.csv"
df = pd.read_csv(file_path).dropna()

# Ensure required columns exist
required_columns = [
    "How do you read the narrator’s emotional vibe in this excerpt?",
    "What’s the big takeaway vibe from this excerpt?",
    "What’s your instant feel from the excerpt’s tone?",
    "‘Accumulation and responsibility’—how does that land with you?",
    "Psychological_State"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Combine selected text fields into a single input feature
df["combined_text"] = df[
    [
        "How do you read the narrator’s emotional vibe in this excerpt?",
        "What’s the big takeaway vibe from this excerpt?",
        "What’s your instant feel from the excerpt’s tone?",
        "‘Accumulation and responsibility’—how does that land with you?"
    ]
].agg(" ".join, axis=1)

# Extract features (X) and target (y)
X = df["combined_text"]
y = df["Psychological_State"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline for text classification
psychological_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train model
psychological_model.fit(X_train, y_train)

# Evaluate model
y_pred = psychological_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Psychological State Model Accuracy: {accuracy:.2f}")

# Save the trained model
model_path = "/workspaces/Psych-Analysis/backend/models/psychological_model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(psychological_model, file)

#load the model to check if it works
with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)
# Test the loaded model
test_text = "I am happy and excited about the future."
predicted_state = loaded_model.predict([test_text])[0]
print(f"Predicted Psychological State for test text: {predicted_state}")
# Confirmation message

print("✅ Psychological State model trained and saved successfully!")
