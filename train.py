import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load your dataset
# Replace this with your actual dataset loading code
# For example:
# df = pd.read_csv('spam.csv')
# X = df['message']
# y = df['label']

# For demonstration, let's create a small sample dataset
X = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "Hello, how are you doing?",
    "URGENT! You have won a 1 week FREE membership",
    "Hi Tom, how's work?",
    "SIX chances to win CASH!",
    "Meeting at 3pm tomorrow"
]
y = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Preprocess the text data
X_transformed = [transform_text(text) for text in X]

# Create and fit the TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_transformed)

# Train the model
model = MultinomialNB()
model.fit(X_tfidf, y)

# Save the vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer have been trained and saved successfully!") 