from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load('linear_regression_model.pkl')

# Define the features used in training
features = ["Cycle", "Time Measured(Sec)", "Voltage Measured(V)", "Current Measured", "Temperature Measured"]

# Home route - renders the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route - handles the form submission and makes predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from the form
        cycle = float(request.form['cycle'])
        time_measured = float(request.form['time_measured'])
        voltage_measured = float(request.form['voltage_measured'])
        current_measured = float(request.form['current_measured'])
        temperature_measured = float(request.form['temperature_measured'])
        
        # Create a DataFrame from the input data
        custom_data = {
            "Cycle": [cycle],  
            "Time Measured(Sec)": [time_measured],  
            "Voltage Measured(V)": [voltage_measured],  
            "Current Measured": [current_measured],  
            "Temperature Measured": [temperature_measured]  
        }
        custom_data_df = pd.DataFrame(custom_data)

        # Ensure columns are in the correct order
        X_custom = custom_data_df[features].values

        # Make predictions using the loaded model
        predictions = loaded_model.predict(X_custom)

        # Render the prediction result in a new HTML page
        return render_template('predict.html', prediction=predictions[0])

if __name__ == '__main__':
    app.run(debug=True)
