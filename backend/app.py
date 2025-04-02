from flask import Flask, request, jsonify, render_template
from training_models import RobustTextClassifier
import joblib
import os

app = Flask(__name__)

# Initialize classifier
classifier = None

def load_classifier():
    global classifier
    model_path = 'robust_text_classifier.pkl'
    if os.path.exists(model_path):
        try:
            classifier = RobustTextClassifier.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            classifier = None
    else:
        print("Model file not found. Please train the model first.")
        classifier = None

# Load classifier when starting the app
load_classifier()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not classifier:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
            
        # Get prediction
        prediction = classifier.predict(text)
        
        # Try to get probabilities if available
        try:
            probas = classifier.predict_proba(text)
            prediction['probabilities'] = probas
        except:
            prediction['probabilities'] = 'Not available'
            
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and file.filename.endswith('.csv'):
            try:
                # Save the file temporarily
                filepath = 'temp_dataset.csv'
                file.save(filepath)
                
                # Initialize and train classifier
                global classifier
                classifier = RobustTextClassifier()
                df = classifier.load_data(filepath)
                classifier.train(df)
                
                # Save the trained model
                classifier.save_model('robust_text_classifier.pkl')
                
                # Clean up
                os.remove(filepath)
                
                return jsonify({'message': 'Model trained and saved successfully'})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400

if __name__ == '__main__':
    app.run(debug=True)