import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "/workspaces/Psych-Analysis/backend/data/sentiment_emotion_psych_dataset_enhanced.csv"
df = pd.read_csv(file_path)

# Select text input and labels
TEXT_COLUMN = "What’s the big takeaway vibe from this excerpt?"  # Choose key text input
SENTIMENT_COLUMN = "Sentiment"
EMOTION_COLUMN = "Emotion"

# Encode labels
sentiment_encoder = LabelEncoder()
emotion_encoder = LabelEncoder()
df["Sentiment_Label"] = sentiment_encoder.fit_transform(df[SENTIMENT_COLUMN])
df["Emotion_Label"] = emotion_encoder.fit_transform(df[EMOTION_COLUMN])

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Create datasets
sentiment_dataset = TextDataset(df[TEXT_COLUMN].tolist(), df["Sentiment_Label"].tolist())
emotion_dataset = TextDataset(df[TEXT_COLUMN].tolist(), df["Emotion_Label"].tolist())

# Split into train and validation sets
train_size = int(0.8 * len(sentiment_dataset))
val_size = len(sentiment_dataset) - train_size

sentiment_train, sentiment_val = random_split(sentiment_dataset, [train_size, val_size])
emotion_train, emotion_val = random_split(emotion_dataset, [train_size, val_size])

# DataLoader
BATCH_SIZE = 8
sentiment_train_loader = DataLoader(sentiment_train, batch_size=BATCH_SIZE, shuffle=True)
sentiment_val_loader = DataLoader(sentiment_val, batch_size=BATCH_SIZE)

emotion_train_loader = DataLoader(emotion_train, batch_size=BATCH_SIZE, shuffle=True)
emotion_val_loader = DataLoader(emotion_val, batch_size=BATCH_SIZE)

# Define training function
def train_model(model, train_loader, val_loader, num_labels, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

        # Validation
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        print(f"Validation Accuracy: {accuracy}")
        print(classification_report(true_labels, predictions))

# Train Sentiment Model
num_sentiment_classes = df["Sentiment_Label"].nunique()
sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_sentiment_classes)
train_model(sentiment_model, sentiment_train_loader, sentiment_val_loader, num_sentiment_classes)

# Train Emotion Model
num_emotion_classes = df["Emotion_Label"].nunique()
emotion_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_emotion_classes)
train_model(emotion_model, emotion_train_loader, emotion_val_loader, num_emotion_classes)

# Save models
sentiment_model.save_pretrained("sentiment_model")
emotion_model.save_pretrained("emotion_model")
print("Models saved successfully.")
