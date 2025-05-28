from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

app = Flask(__name__)
CORS(app)  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    rf_model = joblib.load('random_forest_model.pkl')
    mlp_model = joblib.load('mlp_classifier.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise e

FIELD_MAPPING = {
    "Age": "Age",
    "Sleep Start Time (0-23)": "Sleep Start Time",
    "Sleep End Time (0-23)": "Sleep End Time",
    "Total Sleep Hours": "Total Sleep Hours",
    "Exercise (minutes/day)": "Exercise (mins/day)",
    "Caffeine Intake (mg)": "Caffeine Intake (mg)",
    "Screen Time Before Bed (minutes)": "Screen Time Before Bed (mins)",
    "Work Hours (hours/day)": "Work Hours (hrs/day)",
    "Productivity Score (1-10)": "Productivity Score (1-10)",
    "Mood Score (1-10)": "Mood Score (1-10)",
    "Stress Level (1-10)": "Stress Level (1-10)",
    "Sleep Quality (1-10)": "Sleep Quality (1-10, based on your sleepiness during the day)"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        frontend_data = request.get_json()
        if not frontend_data:
            logger.error("No data received")
            return jsonify({"error": "No data received"}), 400

        logger.info(f"Received data: {frontend_data}")

        model_input = []
        missing_fields = []

        for frontend_key, model_key in FIELD_MAPPING.items():
            if frontend_key not in frontend_data:
                missing_fields.append(frontend_key)
                continue
            try:
                value = float(frontend_data[frontend_key])
                model_input.append(value)
            except (ValueError, TypeError):
                logger.error(f"Invalid value for {frontend_key}: {frontend_data[frontend_key]}")
                return jsonify({"error": f"Invalid value for {frontend_key}"}), 400

        if missing_fields:
            logger.error(f"Missing fields: {missing_fields}")
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        input_array = np.array(model_input).reshape(1, -1)
        logger.info(f"Model input array: {input_array}")

        scaled_input = scaler.transform(input_array)

        rf_pred = rf_model.predict(scaled_input)[0]
        mlp_pred = mlp_model.predict(scaled_input)[0]

        rf_label = label_encoder.inverse_transform([rf_pred])[0]
        mlp_label = label_encoder.inverse_transform([mlp_pred])[0]

        response = {
            "RandomForest": rf_label,
            "NeuralNetwork": mlp_label,
            "status": "success"
        }

        logger.info(f"Returning response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sleep Classifier</title>
    </head>
    <body>
        <h1>Sleep Classifier Backend</h1>
        <p>The classifier backend is running. Use the frontend interface to interact with the service.</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
