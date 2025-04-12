from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import json
import random

# Enhanced training data
training_data = [
    ("I'm so sad today", "sad"),
    ("I feel terrible", "sad"),
    ("This is the worst day", "sad"),
    ("I'm feeling down", "sad"),
    ("I'm so happy right now", "happy"),
    ("This is amazing", "happy"),
    ("I feel great today", "happy"),
    ("What a wonderful day", "happy"),
    ("I'm exhausted", "tired"),
    ("I can't keep my eyes open", "tired"),
    ("I'm burned out", "tired"),
    ("I'm so anxious about this", "anxious"),
    ("I'm freaking out", "anxious"),
    ("I'm so nervous", "anxious"),
    ("This makes me so angry", "angry"),
    ("I'm furious about this", "angry"),
    ("I'm really excited", "excited"),
    ("I can't wait", "excited"),
    ("I'm feeling overwhelmed", "overwhelmed"),
    ("There's too much to do", "overwhelmed"),
    ("I'm okay", "neutral"),
    ("Not bad", "neutral"),
    ("Same as usual", "neutral")
]

# Create the model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
X = [text for text, label in training_data]
y = [label for text, label in training_data]
model.fit(X, y)

def predict_sentiment(text):
    # First check for keyword matches
    with open('comfort_words.json', 'r') as f:
        data = json.load(f)
    
    text_lower = text.lower()
    for sentiment, keywords in data['sentiment_keywords'].items():
        if any(keyword in text_lower for keyword in keywords):
            return sentiment
    
    # If no keyword match, use ML model
    prediction = model.predict([text])[0]
    return prediction

def get_comfort_response(sentiment):
    with open('comfort_words.json', 'r') as f:
        data = json.load(f)
    
    responses = data['responses'].get(sentiment, data['responses']['neutral'])
    return random.choice(responses)