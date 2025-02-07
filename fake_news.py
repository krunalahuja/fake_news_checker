import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib

# Download required NLTK datasets
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Check for missing values
print("Missing values in True.csv:\n", true.isnull().sum())
print("Missing values in Fake.csv:\n", fake.isnull().sum())

# Add labels (0 = Real, 1 = Fake)
true["label"] = 0
fake["label"] = 1

# Combine datasets
data = pd.concat([true, fake], ignore_index=True)

# Display basic info
print(data.info())
print(data.head())

# Drop unnecessary columns
data.drop(columns=["title", "date", "subject"], inplace=True)

# Remove duplicates
if data.duplicated().sum() > 0:
    data.drop_duplicates(inplace=True)

# Define text processing function
def process_text(text):
    """Cleans and preprocesses text."""
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)  # Remove single characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
    text = text.lower()

    words = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words and len(word) > 3]

    return " ".join(words)

# Apply text preprocessing
data["processed_text"] = data["text"].apply(process_text)

# Prepare features and labels
X = data["processed_text"]
y = data["label"].values.ravel()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train model
model = DecisionTreeClassifier()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"{model.__class__.__name__} Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
joblib.dump(model, "models.pkl")
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")




