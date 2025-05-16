import pandas as pd
import joblib
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import neattext.functions as nfx

# Download required NLTK resources
# This is the key change - we're explicitly downloading the correct resource
print("Checking and downloading NLTK resources...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading wordnet...")
    nltk.download('wordnet')


class EmotionDetector:
    """Main emotion detection model class"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.model = None
        self.emotions = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Handle missing/null values
        if pd.isna(text) or text is None:
            return ""
            
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = nfx.remove_urls(text)
        # Remove emails
        text = nfx.remove_emails(text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        # Using direct tokenization approach that doesn't require punkt_tab
        # Just split by spaces as a simple tokenization
        tokens = text.split()
        
        # For more advanced tokenization when punkt works:
        # try:
        #     tokens = nltk.word_tokenize(text)
        # except LookupError:
        #     # Fallback to simple tokenization
        #     print("Warning: NLTK tokenizer not available, using simple tokenization")
        #     tokens = text.split()
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]

        return ' '.join(cleaned_tokens)
        
    def load_data(self, file_path):
        """Load and prepare the emotion dataset with flexible column mapping"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            # Assuming format: text;emotion
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) == 2:
                        data.append({'text': parts[0], 'emotion': parts[1]})
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Please use CSV or TXT files.")

        # Print columns for debugging
        print(f"Columns in dataset: {df.columns.tolist()}")
        
        # Check for common column patterns and map them
        # Case 1: Default format (text, emotion)
        if 'text' in df.columns and 'emotion' in df.columns:
            # Already in the expected format
            pass
            
        # Case 2: Twitter sentiment data (tweet_id, sentiment, content)
        elif all(col in df.columns for col in ['tweet_id', 'sentiment', 'content']):
            print("Detected Twitter sentiment dataset format. Mapping columns...")
            # Map content -> text and sentiment -> emotion
            df['text'] = df['content']
            df['emotion'] = df['sentiment']
            
        # Case 3: Try to infer columns based on content
        else:
            # Look for columns that might contain text (choose the one with longest average length)
            text_candidates = [col for col in df.columns if df[col].dtype == 'object']
            if text_candidates:
                avg_lengths = {col: df[col].astype(str).str.len().mean() for col in text_candidates}
                # Find the column with the longest average text length - likely to be the content
                text_col = max(avg_lengths, key=avg_lengths.get)
                df['text'] = df[text_col]
                print(f"Mapped column '{text_col}' to 'text'")
                
                # Look for columns that might contain sentiment/emotion labels
                # Typical emotion columns have few unique values
                emotion_candidates = [col for col in df.columns if col != text_col and df[col].dtype == 'object' 
                                     and df[col].nunique() < len(df) / 5]  # Heuristic: unique values < 20% of rows
                
                if emotion_candidates:
                    # Choose the one with fewest unique values
                    unique_counts = {col: df[col].nunique() for col in emotion_candidates}
                    emotion_col = min(unique_counts, key=unique_counts.get)
                    df['emotion'] = df[emotion_col]
                    print(f"Mapped column '{emotion_col}' to 'emotion'")
                else:
                    raise ValueError("Could not identify an emotion/sentiment column in the dataset.")
            else:
                raise ValueError("Could not identify a text column in the dataset.")
        
        # Verify required columns now exist
        required_cols = ['text', 'emotion']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' could not be mapped in the dataset.")
        
        # Clean the dataset
        print("Preprocessing text...")
        df['clean_text'] = df['text'].apply(self.preprocess_text)
        
        # Get unique emotions
        self.emotions = df['emotion'].unique()
        print(f"Found {len(self.emotions)} unique emotions/sentiments: {sorted(self.emotions)}")

        return df
        
    def train(self, X_train, y_train, model_type='nb', ngram_range=(1, 2)):
        """Train the emotion detection model"""
        # Create and fit vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Train model
        if model_type == 'nb':
            self.model = MultinomialNB()
        elif model_type == 'lr':
            self.model = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial')
        else:
            raise ValueError("Unsupported model type. Use 'nb' for Naive Bayes or 'lr' for Logistic Regression.")

        self.model.fit(X_train_vec, y_train)
        return self
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model and display results"""
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return accuracy, report, cm
        
    def predict(self, text):
        """Predict emotion for a given text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")

        # Preprocess the text
        clean_text = self.preprocess_text(text)

        # Vectorize and predict
        text_vec = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(text_vec)[0]

        # Get probabilities for all emotions
        probabilities = self.model.predict_proba(text_vec)[0]
        emotion_probs = {emotion: prob for emotion, prob in zip(self.model.classes_, probabilities)}

        return prediction, emotion_probs
        
    def save_model(self, path='emotion_model'):
        """Save the trained model and vectorizer"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Nothing to save.")

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save the model, vectorizer, and emotions list
        joblib.dump(self.model, os.path.join(path, 'model.pkl'))
        joblib.dump(self.vectorizer, os.path.join(path, 'vectorizer.pkl'))
        joblib.dump(self.emotions, os.path.join(path, 'emotions.pkl'))

        return True
        
    def load_model(self, path='emotion_model'):
        """Load a saved model and vectorizer"""
        model_path = os.path.join(path, 'model.pkl')
        vectorizer_path = os.path.join(path, 'vectorizer.pkl')
        emotions_path = os.path.join(path, 'emotions.pkl')

        if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(emotions_path)):
            raise FileNotFoundError(f"Model files not found in {path}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.emotions = joblib.load(emotions_path)

        return self