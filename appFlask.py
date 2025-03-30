from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
model = None

def load_model():
    global model
    model = joblib.load("model.joblib")
    print("Model loaded successfully!")

# Load model when the app starts
@app.before_first_request
def before_first_request():
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature values from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Create input array for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get the species name
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        predicted_species = species_names[np.argmax(prediction[0, 1:])]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        prob_values = [float(p) for p in probabilities[1:]]  # Skip the first value which is just for one-hot encoding
        
        return render_template('result.html', 
                              prediction=predicted_species,
                              sepal_length=sepal_length,
                              sepal_width=sepal_width,
                              petal_length=petal_length,
                              petal_width=petal_width,
                              setosa_prob=f"{prob_values[0]:.2%}",
                              versicolor_prob=f"{prob_values[1]:.2%}",
                              virginica_prob=f"{prob_values[2]:.2%}")
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        # Get JSON data
        data = request.get_json(force=True)
        
        # Extract features
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        
        # Create input array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Get species name
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        predicted_species = species_names[np.argmax(prediction[0, 1:])]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        prob_values = [float(p) for p in probabilities[1:]]
        
        # Prepare response
        response = {
            'prediction': predicted_species,
            'probabilities': {
                'Setosa': f"{prob_values[0]:.4f}",
                'Versicolor': f"{prob_values[1]:.4f}",
                'Virginica': f"{prob_values[2]:.4f}"
            },
            'input_data': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Create an endpoint to get dataset info
@app.route('/dataset')
def dataset_info():
    try:
        # Load the training data
        train_data = pd.read_csv('train_data.csv')
        
        # Get basic stats
        stats = {
            'num_samples': len(train_data),
            'features': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
            'species': ['Setosa', 'Versicolor', 'Virginica'],
            'feature_stats': {
                'sepal_length': {
                    'min': float(train_data['sepal length (cm)'].min()),
                    'max': float(train_data['sepal length (cm)'].max()),
                    'mean': float(train_data['sepal length (cm)'].mean())
                },
                'sepal_width': {
                    'min': float(train_data['sepal width (cm)'].min()),
                    'max': float(train_data['sepal width (cm)'].max()),
                    'mean': float(train_data['sepal width (cm)'].mean())
                },
                'petal_length': {
                    'min': float(train_data['petal length (cm)'].min()),
                    'max': float(train_data['petal length (cm)'].max()),
                    'mean': float(train_data['petal length (cm)'].mean())
                },
                'petal_width': {
                    'min': float(train_data['petal width (cm)'].min()),
                    'max': float(train_data['petal width (cm)'].max()),
                    'mean': float(train_data['petal width (cm)'].mean())
                }
            }
        }
        
        return render_template('dataset.html', stats=stats)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)