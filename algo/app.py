from flask import Flask, render_template, request, jsonify
from sentiment_model import predict_sentiment, get_comfort_response
import json

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_input = data['message']
    
    # Predict sentiment
    sentiment = predict_sentiment(user_input)
    
    # Get appropriate response
    response = get_comfort_response(sentiment)
    
    return jsonify({
        'sentiment': sentiment,
        'response': response
    })

@app.route('/update_phrases', methods=['POST'])
def update_phrases():
    try:
        data = request.get_json()
        phrase_type = data['type']
        new_phrase = data['phrase']
        
        with open('comfort_words.json', 'r+') as f:
            phrases = json.load(f)
            phrases[phrase_type].append(new_phrase)
            f.seek(0)
            json.dump(phrases, f, indent=4)
            f.truncate()
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)