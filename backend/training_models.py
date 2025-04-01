import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import nltk
import spacy
from sklearn.base import clone  # Add this import at the top
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# Download NLTK resources
nltk.download(['stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt'])

class RobustTextClassifier:
    def __init__(self, model_type='svm'):
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        self.label_encoders = {}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.model_type = model_type
        
        # Load spaCy for advanced NLP tasks
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            raise ImportError("spaCy English model not found. Run 'python -m spacy download en_core_web_sm'")
        
        # Enhanced lexicons with fallback defaults
        self._load_lexicons()
        
    def _load_lexicons(self):
        self.positive_words = ['excit', 'happy', 'joy', 'delight', 'love', 'great', 'awesome']
        self.negative_words = ['angry', 'sad', 'hate', 'depress', 'stress', 'worri']
        
        self.emotion_lexicons = {
            'Happy': ['joy', 'excit', 'happy', 'love', 'delight'],
            'Sad': ['sad', 'depress', 'grief', 'sorrow'],
            'Angry': ['angry', 'mad', 'furious', 'rage'],
            'Anxious': ['anxious', 'nervous', 'worri', 'fear'],
            'Calm': ['calm', 'peace', 'serene', 'relax']
        }
        
        self.psych_state_indicators = {
            'Overwhelmed': ['overwhelm', 'too much', 'buried', 'drown', 'can\'t handle'],
            'Purpose-Driven': ['purpose', 'meaning', 'mission', 'why'],
            'Disconnected': ['alone', 'isolate', 'disconnect', 'lonely'],
            'FOMO': ['miss out', 'fomo', 'left out', 'everyone else']
        }

    def preprocess_text(self, text):
        """Advanced text preprocessing with enhanced negation handling"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation for context
        text = re.sub(r'[^a-zA-Z\s\.,!?]', '', text)
        
        # Enhanced negation handling
        negation_patterns = [
            (r'\b(?:not|never|no|none|nobody|nothing|neither|nor|nowhere)\s+(\w+)', r'not_\1'),
            (r"n't\s+(\w+)", r' not_\1'),
            (r'\b(?:rarely|seldom|hardly|scarcely)\s+(\w+)', r'not_\1')
        ]
        
        for pattern, replacement in negation_patterns:
            text = re.sub(pattern, replacement, text)
        
        # POS-aware lemmatization
        doc = self.nlp(text)
        lemmatized = []
        for token in doc:
            if not token.is_punct and not token.is_stop:
                lemma = token.lemma_.lower().strip()
                if lemma:
                    lemmatized.append(lemma)
        
        return ' '.join(lemmatized)

    def load_data(self, filepath):
        """Load and prepare dataset with comprehensive validation"""
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
        
        # Validate required columns
        required_cols = ['Response', 'Sentiment', 'Emotion', 'Psychological_State']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")
            
        # Check for empty responses
        if df['Response'].isnull().any():
            raise ValueError("Dataset contains empty responses")
            
        df['processed_text'] = df['Response'].apply(self.preprocess_text)
        return df
        

    def train(self, df, n_splits=5):
        """Ultimate robust training with flexible label verification"""
        # Load data
        X = df['processed_text']
        y = df[['Sentiment', 'Emotion', 'Psychological_State']].copy()
        
        # Debug: Initial inspection
        print("\nINITIAL DATA INSPECTION:")
        print(f"DataFrame shape: {y.shape}")
        for col in y.columns:
            print(f"\nColumn: {col}")
            print(f"Missing values: {y[col].isna().sum()}")
            print(f"Unique values: {y[col].nunique()}")
            print(f"Sample values: {y[col].unique()[:5]}")
            print(f"Value counts:\n{y[col].value_counts(dropna=False).head()}")
        
        # Convert all labels to strings
        for col in y.columns:
            y[col] = y[col].astype(str)
            
            # Clean strings
            y[col] = y[col].str.strip().str.upper()
            y[col] = y[col].replace({'NAN': 'MISSING', 'NONE': 'MISSING'})
            
            # Merge rare categories
            counts = y[col].value_counts()
            rare_labels = counts[counts < 5].index
            if len(rare_labels) > 0:
                y.loc[y[col].isin(rare_labels), col] = 'OTHER'
        
        # Debug: Post-cleaning
        print("\nPOST-CLEANING INSPECTION:")
        for col in y.columns:
            print(f"\n{col}:")
            print(f"Unique values: {y[col].unique()}")
        
        # Flexible label encoding
        for col in y.columns:
            le = LabelEncoder()
            y[col] = le.fit_transform(y[col])
            self.label_encoders[col] = le
            
            # Verify numeric nature without strict type checking
            if not pd.api.types.is_numeric_dtype(y[col]):
                raise TypeError(f"Non-numeric values in {col} after encoding")
        
        # Final flexible verification
        print("\nFINAL VERIFICATION:")
        valid_types = (int, np.int32, np.int64, np.int8, np.int16)
        for col in y.columns:
            types = {type(x) for x in y[col]}
            print(f"{col} types: {types}")
            
            if not any(issubclass(t, valid_types) for t in types):
                sample = [x for x in y[col] if not isinstance(x, valid_types)][:5]
                raise TypeError(f"Invalid types in {col}. Sample: {sample}")
        
        # Proceed with training
        X_tfidf = self.tfidf.fit_transform(X)
        
        # Initialize model
        self.model = MultiOutputClassifier(
            LinearSVC(class_weight='balanced', dual=False, max_iter=10000)
        )
        
        # Train
        print("\nTRAINING MODEL...")
        self.model.fit(X_tfidf, y)
        
        print("\nSUCCESS! Model trained with verified labels.")
        return self
    def predict_proba(self, text):
        """Get prediction probabilities with proper handling"""
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_text = self.preprocess_text(text)
        text_vector = self.tfidf.transform([processed_text])
        
        try:
            # Try getting probabilities directly
            probas = self.model.predict_proba(text_vector)
            results = {}
            for i, col in enumerate(self.label_encoders.keys()):
                classes = self.label_encoders[col].classes_
                results[col] = {cls: probas[i][0][idx] for idx, cls in enumerate(classes)}
            return results
        except AttributeError:
            # Fallback for models without probability support
            print("Probability not available - using uniform distribution")
            results = {}
            for col in self.label_encoders.keys():
                classes = self.label_encoders[col].classes_
                results[col] = {cls: 1.0/len(classes) for cls in classes}
            return results

    def predict(self, text, confidence_threshold=0.6):
        """Enhanced prediction with better psychological state detection"""
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_text = self.preprocess_text(text)
        text_vector = self.tfidf.transform([processed_text])
        text_lower = text.lower()
        
        # Get base predictions
        class_preds = self.model.predict(text_vector)
        result = {
            col: self.label_encoders[col].inverse_transform([class_preds[0][i]])[0]
            for i, col in enumerate(self.label_encoders.keys())
        }

        # ===== Enhanced Psychological State Detection =====
        psych_state = None
        
        # Check for clear indicators in order of priority
        if any(word in text_lower for word in ['overwhelm', 'too much', 'buried', 'drown']):
            psych_state = 'Overwhelmed'
        elif any(word in text_lower for word in ['purpose', 'meaning', 'mission', 'why']):
            psych_state = 'Purpose-Driven'
        elif any(word in text_lower for word in ['alone', 'isolate', 'disconnect', 'lonely']):
            psych_state = 'Disconnected'
        elif any(word in text_lower for word in ['miss out', 'fomo', 'left out', 'fear of missing']):
            psych_state = 'FOMO'
        elif 'reflect' in text_lower or 'think back' in text_lower:
            psych_state = 'Reflective'
        
        # Only override if we found a clear indicator
        if psych_state:
            result['Psychological_State'] = psych_state

        # ===== Enhanced Emotion Detection =====
        if 'happy' in text_lower or 'excit' in text_lower or 'joy' in text_lower:
            result['Emotion'] = 'Happy'
        elif 'sad' in text_lower or 'depress' in text_lower or 'grief' in text_lower:
            result['Emotion'] = 'Sad'
        elif 'angry' in text_lower or 'mad' in text_lower or 'furious' in text_lower:
            result['Emotion'] = 'Angry'
        elif 'anxious' in text_lower or 'nervous' in text_lower or 'worri' in text_lower:
            result['Emotion'] = 'Anxious'
        elif 'calm' in text_lower or 'peace' in text_lower or 'serene' in text_lower:
            result['Emotion'] = 'Calm'
        elif 'fear' in text_lower or 'scare' in text_lower or 'afraid' in text_lower:
            result['Emotion'] = 'Fear'

        # ===== Enhanced Sentiment Detection =====
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count > negative_count:
            result['Sentiment'] = 'Positive'
        elif negative_count > positive_count:
            result['Sentiment'] = 'Negative'
        # Otherwise keep model's prediction

        return result
    def save_model(self, path):
        """Save all model components with validation"""
        if not self.model:
            raise ValueError("No model to save. Train first.")
            
        save_data = {
            'model': self.model,
            'tfidf': self.tfidf,
            'label_encoders': self.label_encoders,
            'model_type': self.model_type,
            'lexicons': {
                'positive': self.positive_words,
                'negative': self.negative_words,
                'emotion': self.emotion_lexicons,
                'psych_state': self.psych_state_indicators
            }
        }
        joblib.dump(save_data, path)

    @classmethod
    def load_model(cls, path):
        """Load saved model with validation"""
        try:
            data = joblib.load(path)
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
            
        required_keys = ['model', 'tfidf', 'label_encoders']
        if not all(key in data for key in required_keys):
            raise ValueError("Invalid model file format")
            
        classifier = cls(model_type=data.get('model_type', 'svm'))
        classifier.model = data['model']
        classifier.tfidf = data['tfidf']
        classifier.label_encoders = data['label_encoders']
        
        # Restore lexicons if available
        if 'lexicons' in data:
            lexicons = data['lexicons']
            classifier.positive_words = lexicons.get('positive', [])
            classifier.negative_words = lexicons.get('negative', [])
            classifier.emotion_lexicons = lexicons.get('emotion', {})
            classifier.psych_state_indicators = lexicons.get('psych_state', {})
            
        return classifier

if __name__ == "__main__":
    try:
        # Initialize with MLP classifier
        print("Initializing classifier...")
        classifier = RobustTextClassifier(model_type='svm')
        
        # Load and prepare data
        print("Loading data...")
        df = classifier.load_data('data/sentiment_emotion_psych_dataset_enhanced.csv')
        
        # Train model
        print("Training model...")
        classifier.train(df)
        
        # Save model
        print("Saving model...")
        classifier.save_model('robust_text_classifier.pkl')
        
        # Test predictions
        test_cases = [
            "I'm excited about new opportunities",
            "searching for meaning in his past freaked out that i might be losing the real stuff",
            "I feel overwhelmed with all these tasks",
            "This peaceful moment makes me feel serene",
            "I don't like this situation at all",
            "Never been happier in my life!"
        ]
        
        print("\nTest Predictions:")
        for text in test_cases:
            try:
                print(f"\nText: {text}")
                prediction = classifier.predict(text)
                print("Final prediction:", prediction)
                try:
                    probas = classifier.predict_proba(text)
                    print("Probabilities:")
                    for col, vals in probas.items():
                        print(f"  {col}:")
                        for cls, prob in vals.items():
                            print(f"    {cls}: {prob:.3f}")
                except Exception as e:
                    print(f"Could not get probabilities: {str(e)}")
            except Exception as e:
                print(f"Error predicting for text '{text[:30]}...': {str(e)}")
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")