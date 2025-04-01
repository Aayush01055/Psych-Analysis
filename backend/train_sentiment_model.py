import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("/workspaces/Psych-Analysis/backend/data/n1.csv")  

# Ensure required columns exist and drop missing values
if 'Sentiment' not in df.columns:
    raise ValueError("Dataset is missing the 'Sentiment' column.")
df = df.dropna(subset=['Sentiment'])

# Define text columns for feature extraction
text_columns = [
    #column names from the dataset
    'How do you read the narrator’s emotional vibe in this excerpt?',
    'The narrator says memory isn’t always what we witnessed. What’s that make you think about your own memories?',
    'The narrator talks about time molding us. How does that hit you?',
    'When he mentions ‘the end of any likelihood of change,’ what’s your reaction?',
    'The narrator doesn’t miss his schooldays. How do you connect with that?',
    'What’s your instant feel from the excerpt’s tone?',
    '‘Accumulation and responsibility’—how does that land with you?',
    'What’s the big takeaway vibe from this excerpt?'
]

# Combine text columns into a single feature
df['Combined_Text'] = df[text_columns].astype(str).agg(' '.join, axis=1)

# Check class distribution
print("Sentiment class distribution:\n", df['Sentiment'].value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["Combined_Text"], df["Sentiment"], test_size=0.2, random_state=42, stratify=df["Sentiment"]
)

# Create text processing and classification pipeline
sentiment_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.95, min_df=5)),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=500))
])

# Train the model
sentiment_model.fit(X_train, y_train)

# Evaluate performance
y_pred = sentiment_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the model if accuracy is reasonable
if accuracy > 0.65:
    model_path = "/workspaces/Psych-Analysis/backend/models/sentiment_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(sentiment_model, file)
    print(f"Model saved at: {model_path}")
else:
    print("Accuracy is too low. Consider improving dataset quality or model parameters.")
