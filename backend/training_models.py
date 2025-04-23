
import pandas as pd
import numpy as np
import re
import joblib
import warnings
from typing import List, Dict, Any, Optional, Set, Tuple

from sklearn.model_selection import train_test_split # Keep if needed for external evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
# from sklearn.model_selection import StratifiedKFold # Not strictly needed if using integer cv

import nltk
from nltk.corpus import stopwords
import spacy

# Download NLTK resources if not already present (optional, depends if already downloaded)
# You might want to run these downloads outside the script the first time.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet', quiet=True)


class RobustTextClassifier:
    """
    A robust multi-label text classifier for Sentiment, Emotion, and Psychological State.

    Uses TF-IDF vectorization and potentially calibrated LinearSVC within a MultiOutputClassifier.
    Calibration CV folds are adjusted based on data to prevent errors with small classes.
    Includes advanced preprocessing, rule-based overrides for prediction, and model persistence.
    """
    def __init__(self, tfidf_max_features: int = 5000, tfidf_ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initializes classifier components, deferring the final model structure setup to train().

        Args:
            tfidf_max_features: Maximum number of features for TfidfVectorizer.
            tfidf_ngram_range: N-gram range for TfidfVectorizer.
        """
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range
        )
        # Define the base estimator
        self.base_estimator = LinearSVC(
            class_weight='balanced',
            dual='auto', # Automatically chooses based on n_samples/n_features
            max_iter=10000, # Increased iterations for convergence
            C=1.0          # Regularization parameter
            )

        # Defer initialization of potentially calibrated SVC and final model
        self.calibrated_svc: Optional[CalibratedClassifierCV] = None
        self.model: Optional[MultiOutputClassifier] = None

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.stop_words: Set[str] = set(stopwords.words('english'))

        # Load spaCy for advanced NLP tasks (POS-aware lemmatization)
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except IOError:
            print("spaCy English model ('en_core_web_sm') not found.")
            print("Please run: python -m spacy download en_core_web_sm")
            raise ImportError("spaCy 'en_core_web_sm' model not found.")

        # Load predefined lexicons
        self._load_lexicons()

    def _load_lexicons(self):
        """Loads predefined lexicons for rule-based enhancements."""
        # Use lemmatized forms
        self.positive_words: Set[str] = {'excite', 'happy', 'joy', 'delight', 'love', 'great', 'awesome', 'good', 'wonderful', 'fantastic', 'thrill', 'pleased', 'glad'}
        self.negative_words: Set[str] = {'angry', 'sad', 'hate', 'depress', 'stress', 'worry', 'bad', 'terrible', 'awful', 'fear', 'anxious', 'unhappy', 'miserable', 'irritated', 'annoyed'}

        self.emotion_lexicons: Dict[str, List[str]] = {
            'Happy': ['joy', 'excite', 'happy', 'love', 'delight', 'glad', 'pleased', 'wonderful', 'fantastic', 'awesome'],
            'Sad': ['sad', 'depress', 'grief', 'sorrow', 'unhappy', 'miserable', 'down'],
            'Angry': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'resentful'],
            'Anxious': ['anxious', 'nervous', 'worry', 'fear', 'apprehensive', 'stressed', 'tense'],
            'Calm': ['calm', 'peace', 'serene', 'relax', 'tranquil', 'content'],
            'Fear': ['fear', 'scare', 'afraid', 'terrified', 'frightened', 'anxious', 'worry'] # Overlap ok
        }

        self.psych_state_indicators: Dict[str, List[str]] = {
            'Overwhelmed': ['overwhelm', 'too much', 'buried', 'drown', 'can\'t handle', 'swamped'],
            'Purpose-Driven': ['purpose', 'meaning', 'mission', 'why', 'goal', 'driven', 'focus'],
            'Disconnected': ['alone', 'isolate', 'disconnect', 'lonely', 'detached', 'apart'],
            'FOMO': ['miss out', 'fomo', 'left out', 'everyone else'], # Fear Of Missing Out
            'Reflective': ['reflect', 'think back', 'ponder', 'contemplate', 'introspect'],
            'Stressed': ['stress', 'pressure', 'tense', 'strain', 'anxious', 'overwhelm'], # Overlap ok
            'Grateful': ['grateful', 'thankful', 'appreciate', 'blessed']
        }

    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing using spaCy for lemmatization and improved cleaning.

        Args:
            text: The raw input text string.

        Returns:
            The preprocessed text string.
        """
        if not isinstance(text, str):
            warnings.warn(f"Expected string input, got {type(text)}. Converting to empty string.", UserWarning)
            return ""

        text = text.lower()
        text = re.sub(r'[^\w\s\'.,!?]', '', text) # Keep word chars, whitespace, apostrophe, basic punctuation

        # Enhanced negation handling
        negation_patterns = [
            (r'\b(?:not|never|no|none|nobody|nothing|neither|nor|nowhere)\s+(\w+)', r'not_\1'),
            (r"n't\s+(\w+)", r' not_\1'),
            (r'\b(?:rarely|seldom|hardly|scarcely)\s+(\w+)', r'not_\1')
        ]
        for pattern, replacement in negation_patterns:
            text = re.sub(pattern, replacement, text)

        # spaCy Processing for Lemmatization & Stop Word Removal
        doc = self.nlp(text)
        lemmatized_tokens = []
        for token in doc:
            if not token.is_punct and not token.is_stop and token.lemma_.strip():
                lemmatized_tokens.append(token.lemma_.lower())

        return ' '.join(lemmatized_tokens)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads and prepares dataset from a CSV file with validation.

        Args:
            filepath: Path to the CSV file.

        Returns:
            A pandas DataFrame with raw and processed text.

        Raises:
            ValueError: If the file cannot be loaded, required columns are missing,
                        or the 'Response' column contains null values.
        """
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise ValueError(f"Data file not found at: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load data from {filepath}: {str(e)}")

        required_cols = ['Response', 'Sentiment', 'Emotion', 'Psychological_State']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        if df['Response'].isnull().any():
            num_null = df['Response'].isnull().sum()
            warnings.warn(f"Dataset contains {num_null} empty 'Response' values. Filling with empty string.", UserWarning)
            df['Response'] = df['Response'].fillna('')

        print("Preprocessing text data...")
        df['processed_text'] = df['Response'].apply(self.preprocess_text)
        print("Preprocessing complete.")
        return df

    def train(self, df: pd.DataFrame, min_freq_for_label: int = 5, default_calibration_cv: int = 3):
        """
        Trains the multi-output classifier, dynamically adjusting calibration CV folds.

        Performs label cleaning, encoding, TF-IDF vectorization, calculates minimum
        class frequencies to adjust calibration CV folds, initializes the model structure
        (potentially using CalibratedClassifierCV), and fits the model.
        Prints a classification report on the training data after completion.

        Args:
            df: DataFrame containing 'processed_text' and label columns.
            min_freq_for_label: Minimum frequency for a label category to be kept;
                                others are merged into 'OTHER'.
            default_calibration_cv: The desired number of folds for probability calibration (e.g., 3 or 5).
                                    This will be reduced if data doesn't support it.

        Returns:
            The trained classifier instance (self).

        Raises:
            ValueError: If target columns are missing, encoding fails, or if calibration
                        is requested but impossible due to class counts < 2.
        """
        target_cols = ['Sentiment', 'Emotion', 'Psychological_State']
        if not all(col in df.columns for col in target_cols):
            raise ValueError(f"DataFrame must contain columns: {target_cols}")

        X = df['processed_text']
        y_raw = df[target_cols].copy()

        print("\n--- Label Cleaning and Encoding ---")
        y_encoded = pd.DataFrame(index=y_raw.index)
        self.label_encoders = {} # Reset encoders for fresh training

        for col in target_cols:
            print(f"\nProcessing column: {col}")
            y_col = y_raw[col].fillna('MISSING').astype(str)
            y_col = y_col.str.strip().str.upper()
            y_col = y_col.replace({'NONE': 'MISSING', 'NAN': 'MISSING', '': 'MISSING'})

            counts = y_col.value_counts()
            rare_labels = counts[counts < min_freq_for_label].index
            if not rare_labels.empty:
                print(f"  Merging {len(rare_labels)} rare labels (count < {min_freq_for_label}) into 'OTHER'.")
                y_col[y_col.isin(rare_labels)] = 'OTHER'

            print(f"  Unique values after cleaning: {y_col.nunique()}")
            print(f"  Sample cleaned values: {np.unique(y_col)[:10]}")

            le = LabelEncoder()
            try:
                y_encoded[col] = le.fit_transform(y_col)
                self.label_encoders[col] = le
                print(f"  Successfully encoded. Classes: {list(le.classes_)}")
            except Exception as e:
                 raise ValueError(f"Error encoding column '{col}'. Check data. Original error: {e}")

            if not pd.api.types.is_integer_dtype(y_encoded[col]):
                 raise TypeError(f"Encoding failed for {col}: Resulting dtype is not integer ({y_encoded[col].dtype}).")

        # --- Calculate Minimum Class Frequency for Calibration CV ---
        print("\n--- Checking Minimum Class Frequencies for Calibration ---")
        min_samples_overall = float('inf')
        problematic_cols_info = []

        for col in target_cols:
             counts = y_encoded[col].value_counts()
             if counts.empty:
                 min_in_col = 0 # Handle case where a column might become empty after processing
             else:
                 min_in_col = counts.min()

             min_samples_overall = min(min_samples_overall, min_in_col)
             if min_in_col < 2:
                 problematic_cols_info.append(f"'{col}' (min count: {min_in_col})")

        if min_samples_overall < 2:
             # Option 1: Raise error (Safer - forces user to fix data/params)
             raise ValueError(f"Calibration requires >= 2 samples per class for CV. "
                              f"Columns with < 2 samples in a class: {problematic_cols_info}. "
                              f"Increase `min_freq_for_label` or ensure data quality.")
             # Option 2: Warn and disable calibration (Alternative - allows running but no probabilities)
             # print(f"Warning: Calibration requires >= 2 samples per class. "
             #       f"Columns with < 2 samples: {problematic_cols_info}. "
             #       f"Disabling calibration for this training run.")
             # min_samples_overall = 0 # Flag to disable

        # Determine the number of CV folds possible
        n_splits_for_calib = 0
        if min_samples_overall >= 2:
            n_splits_for_calib = min(default_calibration_cv, min_samples_overall)
            if n_splits_for_calib < default_calibration_cv:
                print(f"Warning: Minimum class count ({min_samples_overall}) is less than requested "
                      f"CV folds ({default_calibration_cv}). Reducing calibration CV folds to {n_splits_for_calib}.")
        else:
            # This case should be handled by the ValueError above, but as a fallback:
            print("Warning: Cannot perform calibration CV (minimum class count < 2).")


        print(f"Using {n_splits_for_calib if n_splits_for_calib >= 2 else 'no'} folds for calibration CV.")

        print("\n--- Feature Extraction (TF-IDF) ---")
        X_tfidf = self.tfidf.fit_transform(X)
        print(f"TF-IDF matrix shape: {X_tfidf.shape}")

        # --- Initialize Model Structure (Dynamically based on possibility of calibration) ---
        print("\n--- Initializing and Training Multi-Output Classifier ---")
        final_estimator = None
        if n_splits_for_calib >= 2:
             # Use calibration if possible
             print(f"Initializing CalibratedClassifierCV with cv={n_splits_for_calib}.")
             self.calibrated_svc = CalibratedClassifierCV(
                 self.base_estimator,
                 cv=n_splits_for_calib,
                 method='isotonic' # Or 'sigmoid'
             )
             final_estimator = self.calibrated_svc
        else:
             # Fallback: Use base estimator without calibration
             print("Initializing with base LinearSVC (no calibration). Probabilities may be unreliable.")
             final_estimator = self.base_estimator # Use the raw SVC

        self.model = MultiOutputClassifier(final_estimator)

        # --- Fit the model ---
        try:
            self.model.fit(X_tfidf, y_encoded)
        except Exception as e:
            raise RuntimeError(f"Model training failed during fit: {str(e)}")

        print("\n--- Training Complete ---")

        # --- Evaluate on Training Data (Basic Check) ---
        print("\n--- Classification Report (on Training Data) ---")
        print("Note: Evaluating on the training set shows fit but may overestimate performance.")
        try:
            y_pred_encoded = self.model.predict(X_tfidf)

            print("\nDetailed Report per Label:")
            # Use .iloc for reliable integer-based indexing if y_encoded columns change order
            y_encoded_np = y_encoded.values

            for i, col in enumerate(target_cols):
                le = self.label_encoders[col]
                target_names = [str(cls) for cls in le.classes_]
                # Ensure correct slicing for multi-output prediction array
                report = classification_report(
                    y_encoded_np[:, i],        # True labels for this column
                    y_pred_encoded[:, i],     # Predicted labels for this column
                    target_names=target_names,
                    zero_division=0
                )
                print(f"\nClassification Report for: {col}\n")
                print(report)

        except Exception as e:
            print(f"\nCould not generate classification report: {str(e)}")

        return self

    def predict_proba(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Predicts class probabilities for each label. Handles cases with/without calibration.

        Args:
            text: The input text string.

        Returns:
            A dictionary {label: {class: probability}}. If calibration was disabled
            during training, probabilities might be uniform or based on decision function.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.model or not self.label_encoders or not hasattr(self.tfidf, 'vocabulary_'):
             raise ValueError("Model not trained or TF-IDF not fitted. Call train() first.")


        processed_text = self.preprocess_text(text)
        text_vector = self.tfidf.transform([processed_text])

        results = {}
        try:
             # Check if the fitted estimator supports predict_proba (Calibrated should, raw SVC might not)
             if hasattr(self.model.estimators_[0], 'predict_proba'):
                 probas_list = self.model.predict_proba(text_vector)
                 for i, col in enumerate(self.label_encoders.keys()):
                     le = self.label_encoders[col]
                     class_probas = probas_list[i][0] # Shape (1, n_classes)
                     results[col] = {cls: prob for cls, prob in zip(le.classes_, class_probas)}
             else:
                 # Fallback if predict_proba is not available (e.g., raw LinearSVC used)
                 warnings.warn("Predict_proba not available for the fitted estimator (calibration might have been disabled). Returning uniform distribution.", RuntimeWarning)
                 for col, le in self.label_encoders.items():
                     num_classes = len(le.classes_)
                     results[col] = {cls: 1.0 / num_classes for cls in le.classes_}
                 # Alternative Fallback (more complex): Use decision_function if available
                 # try:
                 #     dec_func_list = self.model.decision_function(text_vector)
                 #     # Convert decision function scores to pseudo-probabilities (e.g., softmax)
                 #     # This part requires careful implementation.
                 # except AttributeError:
                 #     # Ultimate fallback: uniform
                 #     pass


        except AttributeError as ae:
             # Catch cases where model structure is unexpected or methods missing
             raise RuntimeError(f"Error during probability prediction - model structure issue? {ae}")
        except Exception as e:
             raise RuntimeError(f"Unexpected error during probability prediction: {str(e)}")

        return results


    def predict(self, text: str) -> Dict[str, str]:
        """
        Predicts the most likely class for each label, with rule-based overrides.

        Args:
            text: The input text string.

        Returns:
            A dictionary {label: predicted_class_name}.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.model or not self.label_encoders or not hasattr(self.tfidf, 'vocabulary_'):
             raise ValueError("Model not trained or TF-IDF not fitted. Call train() first.")


        processed_text = self.preprocess_text(text)
        text_vector = self.tfidf.transform([processed_text])

        # Get base predictions from the model
        class_preds_encoded = self.model.predict(text_vector) # Shape (1, n_outputs)

        # Decode base predictions
        result = {}
        for i, col in enumerate(self.label_encoders.keys()):
             le = self.label_encoders[col]
             encoded_pred = class_preds_encoded[0, i]
             try:
                 result[col] = le.inverse_transform([encoded_pred])[0]
             except IndexError:
                  warnings.warn(f"Prediction index {encoded_pred} out of bounds for label '{col}' with classes {le.classes_}. Using first class as fallback.", RuntimeWarning)
                  result[col] = le.classes_[0] if len(le.classes_) > 0 else "UNKNOWN"


        # === Rule-Based Overrides ===
        text_lower = text.lower()
        processed_tokens = set(processed_text.split()) # Use processed tokens for lemmatized checks

        # --- Psych State Override ---
        detected_psych_state = None
        for state, keywords in self.psych_state_indicators.items():
            if any(keyword in text_lower for keyword in keywords): # Check original text for phrases
                 if any(lemma in processed_tokens for lemma in keywords): # Check lemmas too
                     detected_psych_state = state.upper() # Match label encoder format
                     break
        if detected_psych_state and 'Psychological_State' in result:
             # Only override if the detected state is a valid class
             if detected_psych_state in self.label_encoders['Psychological_State'].classes_:
                 result['Psychological_State'] = detected_psych_state
             else:
                  warnings.warn(f"Rule detected psych state '{detected_psych_state}' not in trained classes. Keeping model prediction.", UserWarning)


        # --- Emotion Override ---
        detected_emotion = None
        for emotion, keywords in self.emotion_lexicons.items():
             if any(keyword in text_lower for keyword in keywords):
                  if any(lemma in processed_tokens for lemma in keywords):
                      detected_emotion = emotion.upper()
                      break
        if detected_emotion and 'Emotion' in result:
             if detected_emotion in self.label_encoders['Emotion'].classes_:
                 result['Emotion'] = detected_emotion
             else:
                  warnings.warn(f"Rule detected emotion '{detected_emotion}' not in trained classes. Keeping model prediction.", UserWarning)


        # --- Sentiment Override (Lexicon Count on Processed Text) ---
        if 'Sentiment' in result:
            positive_count = sum(1 for word in self.positive_words if word in processed_tokens)
            negative_count = sum(1 for word in self.negative_words if word in processed_tokens)
            # Adjust counts for negations
            negative_count += sum(1 for word in self.positive_words if f"not_{word}" in processed_tokens)
            positive_count += sum(1 for word in self.negative_words if f"not_{word}" in processed_tokens)

            override_sentiment = None
            if positive_count > negative_count:
                override_sentiment = 'POSITIVE'
            elif negative_count > positive_count:
                override_sentiment = 'NEGATIVE'

            if override_sentiment and override_sentiment in self.label_encoders['Sentiment'].classes_:
                result['Sentiment'] = override_sentiment
            elif override_sentiment:
                 warnings.warn(f"Rule detected sentiment '{override_sentiment}' not in trained classes. Keeping model prediction.", UserWarning)

        return result

    def save_model(self, path: str):
        """
        Saves the trained model and associated components to a file.

        Args:
            path: The file path (typically .pkl) to save the model to.

        Raises:
            ValueError: If the model has not been trained yet.
            IOError: If saving fails.
        """
        if not self.model or not self.label_encoders or not hasattr(self.tfidf, 'vocabulary_') or not self.tfidf.vocabulary_:
            raise ValueError("Model not trained or essential components missing. Train first.")

        save_data = {
            'model': self.model,
            'tfidf': self.tfidf,
            'label_encoders': self.label_encoders,
            'lexicons': { # Save lexicons used during training/prediction
                'positive': self.positive_words,
                'negative': self.negative_words,
                'emotion': self.emotion_lexicons,
                'psych_state': self.psych_state_indicators
            }
        }
        try:
            joblib.dump(save_data, path)
            print(f"Model successfully saved to {path}")
        except Exception as e:
            raise IOError(f"Failed to save model to {path}: {str(e)}")

    @classmethod
    def load_model(cls, path: str):
        """
        Loads a trained model and its components from a file.

        Args:
            path: The file path where the model was saved.

        Returns:
            An instance of RobustTextClassifier with loaded components.

        Raises:
            ValueError: If the file is not found, is corrupted, or is missing required components.
        """
        try:
            data = joblib.load(path)
        except FileNotFoundError:
            raise ValueError(f"Model file not found at: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}. Error: {str(e)}")

        required_keys = ['model', 'tfidf', 'label_encoders']
        if not all(key in data for key in required_keys):
            raise ValueError(f"Invalid model file format. Missing keys: {[k for k in required_keys if k not in data]}")

        # Create a new instance - TF-IDF params are loaded
        classifier = cls()
        classifier.model = data['model']
        classifier.tfidf = data['tfidf']
        classifier.label_encoders = data['label_encoders']

        # Restore lexicons if available, otherwise use defaults
        if 'lexicons' in data:
            lexicons = data['lexicons']
            classifier.positive_words = lexicons.get('positive', classifier.positive_words)
            classifier.negative_words = lexicons.get('negative', classifier.negative_words)
            classifier.emotion_lexicons = lexicons.get('emotion', classifier.emotion_lexicons)
            classifier.psych_state_indicators = lexicons.get('psych_state', classifier.psych_state_indicators)
        else:
             warnings.warn("Loaded model file does not contain lexicons. Using default lexicons.", UserWarning)
             # Defaults are already loaded by __init__

        # Infer base_estimator from loaded model if possible (for consistency, though not strictly needed for predict/save)
        try:
            # Assumes MultiOutputClassifier stores estimators in .estimators_
            first_estimator = classifier.model.estimators_[0]
            if isinstance(first_estimator, CalibratedClassifierCV):
                 classifier.base_estimator = first_estimator.estimator # Get the SVC from inside Calibrated
                 classifier.calibrated_svc = first_estimator
            else:
                 classifier.base_estimator = first_estimator # Assumes it's the raw SVC
        except (AttributeError, IndexError):
             warnings.warn("Could not reliably determine base estimator from loaded model structure.")
             # classifier.base_estimator remains the default LinearSVC() from __init__


        print(f"Model successfully loaded from {path}")
        return classifier

# --- Main Execution Block ---
if __name__ == "__main__":
    # Adjust paths as needed
    DATASET_PATH = 'data/sentiment_emotion_psych_dataset_enhanced.csv'
    MODEL_SAVE_PATH = 'robust_text_classifier_v3.pkl'

    try:
        # Initialize classifier
        print("Initializing classifier...")
        classifier = RobustTextClassifier(tfidf_max_features=5000)

        # Load and prepare data
        print(f"\nLoading data from {DATASET_PATH}...")
        df = classifier.load_data(DATASET_PATH)

        # Train model
        # Adjust min_freq_for_label if you still get errors or want to merge more
        # default_calibration_cv can be adjusted (e.g., 5 if data supports it)
        print("\nStarting model training...")
        classifier.train(df, min_freq_for_label=10, default_calibration_cv=3)

        # Save model
        print(f"\nSaving model to {MODEL_SAVE_PATH}...")
        classifier.save_model(MODEL_SAVE_PATH)

        # --- Test Predictions with Saved Model ---
        print("\n--- Loading and Testing Saved Model ---")
        loaded_classifier = RobustTextClassifier.load_model(MODEL_SAVE_PATH)

        test_cases = [
            "I'm so incredibly excited about the new opportunities opening up!",
            "Feeling completely overwhelmed with all these tasks and deadlines.",
            "He was searching for meaning in his past, freaked out that he might be losing the real stuff.",
            "This peaceful moment by the lake makes me feel utterly serene and calm.",
            "I don't like this situation at all, it makes me quite anxious.",
            "Never been happier in my entire life! This is wonderful.",
            "She felt isolated and disconnected from her friends after moving.",
            "I'm not sad, just feeling a bit reflective today.",
            "The pressure is mounting, I'm feeling really stressed out.",
            "What's the purpose of all this struggle?",
            "It wasn't a bad experience, actually quite enjoyable.",
            "Feeling grateful for the support I received.",
            "This is just okay, neither good nor bad.", # Neutral test
            "I feel nothing.", # Ambiguous
        ]

        print("\nTest Predictions (using loaded model):")
        for text in test_cases:
            print("-" * 20)
            print(f"Text: {text}")
            try:
                prediction = loaded_classifier.predict(text)
                print(" Prediction:", prediction)

                probas = loaded_classifier.predict_proba(text)
                print(" Probabilities:")
                for col, vals in probas.items():
                    sorted_probas = sorted(vals.items(), key=lambda item: item[1], reverse=True)
                    print(f"   {col}:")
                    # Only show top N probabilities for brevity if needed
                    for cls, prob in sorted_probas[:5]:
                        print(f"     {cls}: {prob:.4f}")

            except Exception as e:
                print(f" Error predicting for text '{text[:50]}...': {str(e)}")

    except ImportError as e:
         print(f"\nImport Error: {e}. Ensure required libraries/models (spaCy) are installed.")
    except ValueError as e:
         print(f"\nConfiguration or Data Error: {e}")
    except RuntimeError as e:
         print(f"\nRuntime Error during training or prediction: {e}")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected fatal error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
