from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load model metrics
with open('model_metrics.json', 'r') as f:
    model_metrics = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from POST request
        features = [float(x) for x in request.form.values()]
        # Convert to DataFrame
        features_df = pd.DataFrame([features], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        # Make prediction
        prediction = model.predict(features_df)
        # Prepare response
        result = f'Predicted class: {prediction[0]}'
        return render_template('index.html', prediction_text=result, metrics=model_metrics)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', metrics=model_metrics)

if __name__ == "__main__":
    app.run(debug=True)

