import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "/workspaces/Psych-Analysis/backend/data/n1.csv"  # Update if needed
df = pd.read_csv(file_path)

# Identify emotion column
emotion_column = "Emotion"  # Predefined as per dataset

# Combine multiple text columns to create a more comprehensive input feature
text_columns = [
    "How do you read the narrator’s emotional vibe in this excerpt?",
    "The narrator says memory isn’t always what we witnessed. What’s that make you think about your own memories?",
    "The narrator talks about time molding us. How does that hit you?",
    "When he mentions ‘the end of any likelihood of change,’ what’s your reaction?",
    "The narrator doesn’t miss his schooldays. How do you connect with that?",
    "What’s your instant feel from the excerpt’s tone?",
    "‘Accumulation and responsibility’—how does that land with you?",
    "What’s the big takeaway vibe from this excerpt?"
]

# Fill missing values with empty strings and concatenate text columns
df["Combined_Text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

# Drop missing values in target column
df = df.dropna(subset=[emotion_column])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["Combined_Text"], df[emotion_column], test_size=0.2, random_state=42)

# Create pipeline with TF-IDF and Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Emotion Model Accuracy: {accuracy:.2f}")

# Save the trained model
model_path = "/workspaces/Psych-Analysis/backend/models/emotion_model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Emotion model saved successfully at: {model_path}")
