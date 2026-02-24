from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler with proper path handling
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../Model/Model_File/model.pkl")
scaler_path = os.path.join(base_dir, "../Model/Model_File/scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(f"Model loaded from: {model_path}")
print(f"Scaler loaded from: {scaler_path}")


@app.route('/')
def home():
    return "Student Performance API Running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        G1 = data['G1']
        G2 = data['G2']
        studytime = data['studytime']
        failures = data['failures']
        absences = data['absences']

        input_data = np.array([[G1, G2, studytime, failures, absences]])

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        return jsonify({
            "Predicted_G3": float(prediction[0]),
            "success": True
        })
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)