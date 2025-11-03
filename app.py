import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# 1. Initialize the Flask app
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing) to allow
# our HTML file to make requests to this server
CORS(app) 

# 2. Load our saved model pipeline
model_filename = 'house_rent_predictor.joblib'
try:
    model = joblib.load(model_filename)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found. Please run save_model.py first.")
    exit()

# 3. Define the column order our model expects
# THIS MUST BE THE EXACT SAME ORDER AS IN 'save_model.py'
MODEL_COLUMNS = [
    'BHK', 'Size', 'Bathroom', 'Floor_Num', 'Total_Floors',
    'Is_Top_Floor', 'Is_Ground_Floor', 'Area Type', 'City',
    'Furnishing Status', 'Tenant Preferred', 'Area_Locality_Simple'
]

# 4. Define the homepage route (optional, but good practice)
@app.route('/')
def home():
    # This will just serve the 'index.html' file we already made
    # Make sure 'index.html' is in a folder named 'templates'
    # or just return a simple message.
    return "House Rent Predictor API is running!"

# 5. Define the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data sent from the website
        data = request.get_json()
        
        # --- Create a DataFrame from the input data ---
        # This is the most important step!
        # We create a single-row DataFrame with the columns
        # in the exact order our model was trained on.
        
        # Calculate engineered features
        data['Is_Top_Floor'] = 1 if data['Floor_Num'] == data['Total_Floors'] else 0
        data['Is_Ground_Floor'] = 1 if data['Floor_Num'] == 0 else 0
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Reorder columns to match the model's training
        input_df = input_df[MODEL_COLUMNS]

        # --- Make the prediction ---
        prediction_log = model.predict(input_df)
        
        # 'prediction_log' is an array, so get the first element
        rent_log = prediction_log[0]
        
        # Reverse the log transform (log1p -> expm1)
        rent_actual = np.expm1(rent_log)
        
        # Return the prediction as JSON
        return jsonify({
            'success': True,
            'predicted_rent': round(rent_actual, 2)
        })

    except Exception as e:
        # Handle any errors
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 6. Run the app
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000/
    print("Starting Flask server... Access it at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
