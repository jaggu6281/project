from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("Error: model or scaler file not found. Make sure 'random_forest_model.pkl' and 'scaler.pkl' are in the same directory.")
    exit()

# Define the feature names in the correct order as used during training
# This list must exactly match X.columns from your training data
feature_names = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance\t', 'Previous qualification',
    'Previous qualification (grade)', 'Nacionality', "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    'Admission grade', 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
    'Inflation rate', 'GDP'
]

# Define target mapping (assuming the order from LabelEncoder based on previous cells)
# 0: Dropout, 1: Graduate, 2: Enrolled
target_mapping = {
    0: 'Dropout',
    1: 'Enrolled',
    2: 'Graduate'
}

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_values = []
        for feature in feature_names:
            # Handle potential KeyError if a field is missing, though HTML should prevent this
            input_values.append(float(data[feature]))

        # Create a DataFrame with the correct feature names
        input_df = pd.DataFrame([input_values], columns=feature_names)

        # Scale the input data using the fitted scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction_encoded = model.predict(scaled_input)[0]
        prediction_label = target_mapping.get(prediction_encoded, 'Unknown')

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
    