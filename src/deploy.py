from flask import Flask, request, jsonify
import pickle
from preprocessor import EmailPreprocessor

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
def load_model(model_path='spam_classifier_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Initialize preprocessor
preprocessor = EmailPreprocessor()

# Load model
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for spam prediction
    
    Expected request format:
    {
        "text": "Email text here..."
    }
    
    Returns:
    {
        "prediction": 0 or 1 (0 for ham, 1 for spam),
        "probability": float (spam probability),
        "status": "success" or "error"
    }
    """
    try:
        # Get email text from request
        data = request.json
        email_text = data.get('text', '')
        
        if not email_text:
            return jsonify({
                'status': 'error',
                'message': 'No email text provided'
            }), 400
        
        # Preprocess and transform the text
        X = preprocessor.transform([email_text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # Probability of spam class
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'probability': float(probability),
            'spam': bool(prediction == 1),
            'confidence': float(probability if prediction == 1 else 1 - probability)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)