from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the model from the pkl file
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Fetch the input JSON data

    # Extract visitors and rating from the input data
    visitors = data.get('visitors')
    rating = data.get('rating')

    # Ensure visitors and rating are provided
    if visitors is None or rating is None:
        return jsonify({'error': 'Missing input data'}), 400

    # Prepare the input for the model (assuming it expects a 2D array)
    input_features = np.array([[visitors, rating]])

    # Make a prediction
    prediction = model.predict(input_features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
