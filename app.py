from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the trained model and preprocessing objects
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    label_encoders = None

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoders is None:
        return jsonify({'success': False, 'error': 'Model not loaded correctly'})

    try:
        # Parse JSON data from the request body
        data = request.get_json()
        print("Received data:", data)  # Debugging: print incoming data

        # Parse blood pressure if it's a string
        if isinstance(data['bloodPressure'], str):
            try:
                systolic, diastolic = map(int, data['bloodPressure'].split('/'))
                blood_pressure = systolic  # You might want to modify this based on your model's requirements
            except:
                return jsonify({'success': False, 'error': 'Invalid blood pressure format'})
        else:
            blood_pressure = data['bloodPressure']

        # Check if all required fields are present
        required_fields = [
            'name', 'gender', 'age', 'occupation', 'bloodPressure', 
            'sleepDuration', 'weight', 'height', 'bmiCategory', 
            'physicalActivityLevel', 'dailySteps', 'heartRate', 
            'qualityOfSleep', 'stressLevel'
        ]
        
        if not all(key in data for key in required_fields):
            return jsonify({'success': False, 'error': 'Missing required data fields'})

        # Process and encode categorical data as needed
        try:
            gender_encoded = label_encoders['Gender'].transform([data['gender']])[0]
            occupation_encoded = label_encoders['Occupation'].transform([data['occupation']])[0]
            bmi_category_encoded = label_encoders['BMI Category'].transform([data['bmiCategory']])[0]
            physical_activity_level_encoded = label_encoders['Physical Activity Level'].transform([data['physicalActivityLevel']])[0]
            quality_of_sleep_encoded = label_encoders['Quality of Sleep'].transform([data['qualityOfSleep']])[0]
        except KeyError as e:
            return jsonify({'success': False, 'error': f'Error encoding categorical data: {str(e)}'})

        # Prepare the input features for the model
        input_features = np.array([
            data['age'], gender_encoded, occupation_encoded, bmi_category_encoded, 
            physical_activity_level_encoded, data['dailySteps'], data['sleepDuration'], 
            data['heartRate'], quality_of_sleep_encoded, data['stressLevel'], 
            blood_pressure, data['weight'], data['height']
        ], dtype=float).reshape(1, -1)

        # Scale numerical features
        input_features[:, [0, 5, 6, 7, 10, 11]] = scaler.transform(input_features[:, [0, 5, 6, 7, 10, 11]])
        print(input_features)

        # Make prediction
        prediction = model.predict(input_features)
        prediction_label = label_encoders['Sleep Disorder'].inverse_transform(prediction)[0]
        
        return jsonify({'success': True, 'prediction': prediction_label})

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'success': False, 'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
