from flask import Flask, request, jsonify
import pandas as pd
from preprocess import preprocess_text
from sentiment_analysis import predict_sentiment
from emotion_analysis import predict_emotion
from topic_modeling import extract_topics

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    df = pd.read_csv(file)
    
    df['Processed_Response'] = df['Response'].apply(preprocess_text)
    df['Sentiment'] = df['Processed_Response'].apply(predict_sentiment)
    df['Emotion'] = df['Processed_Response'].apply(predict_emotion)
    topics = extract_topics(df['Processed_Response'].tolist())

    return jsonify({
        "results": df[['Response', 'Sentiment', 'Emotion']].to_dict(orient="records"),
        "topics": topics
    })

if __name__ == '__main__':
    app.run(debug=True)
