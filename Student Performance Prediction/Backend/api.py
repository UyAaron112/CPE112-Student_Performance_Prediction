from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("../Model/Model_File/model.pkl")
scaler = joblib.load("../Model/Model_File/scaler.pkl")


@app.route('/')
def home():
    return "Student Performance API Running"


@app.route('/predict', methods=['POST'])
def predict():

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
        "Predicted_G3": float(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)