from flask import Flask, request, render_template, jsonify
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK datasets
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("models.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the input text from the form
        input_text = request.form["text"]

        # Preprocess the input text
        processed_text = process_text(input_text)

        # Convert the processed text into numerical features
        text_tfidf = vectorizer.transform([processed_text])

        # Make a prediction
        prediction = model.predict(text_tfidf)

        # Map prediction to label
        result = "Fake" if prediction[0] == 1 else "Real"

        # Return the result
        return render_template("index.html", prediction_text=f"The news is {result}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)






