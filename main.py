from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

# Inisialisasi Flask
app = Flask(__name__)

# Memuat model
model = tf.keras.models.load_model('lstm_model.h5')

def forecast(data, column_name, years_to_forecast=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[[column_name]])
    
    # Set up last_data to match the model's input shape
    n_timesteps = model.input_shape[1]
    last_data = scaled_data[-n_timesteps:].reshape(1, n_timesteps, 1)

    predictions = []
    for _ in range(years_to_forecast):
        prediction = model.predict(last_data)
        
        # If prediction is 2D, reshape it to add an extra dimension
        if prediction.ndim == 2:
            prediction = prediction.reshape(1, 1, -1)
        
        # Select the first feature if there are multiple
        prediction = prediction[:, :, 0].reshape(1, 1, 1)

        predictions.append(prediction[0, 0, 0])  # Store only the forecasted value
        last_data = np.append(last_data[:, 1:, :], prediction, axis=1)

    # Inverse scale the predictions
    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_values.flatten()





# Memuat data dari file JSON
with open('data_dana_desa.json') as f:
    data = json.load(f)

# Membuat DataFrame dari data
df = pd.DataFrame(data)

@app.route('/forecast', methods=['POST'])
def forecast_api():
    data = request.json
    column_name = data.get('column_name')
    years_to_forecast = data.get('years_to_forecast', 1)

    predicted_value = forecast(df, column_name, years_to_forecast)
    
    # Convert the NumPy array to a list for JSON serialization
    return jsonify({'predicted_value': predicted_value.tolist()})


# Menjalankan API
if __name__ == '__main__':
    app.run(debug=True)
