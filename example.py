import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#input_features: list of feature names (e.g., ["temperature", "radiation", ...])
#X should be a 2D NumPy array of shape (n_samples, len(input_features))
#y should be a 1D NumPy array of shape (n_samples,) with energy production in kWh

# Sample Data Preparation (Replace with your actual data)
# Assuming `weather_data` is a DataFrame with columns: 'temperature', 'diffuse_radiation', 'direct_radiation', etc.
# Assuming 'energy_data' is the target column for energy production (e.g., kWh)

# Load your data (replace this with actual loading code)
weather_data = pd.DataFrame({
    'temperature': np.random.rand(1000),  # Replace with actual data
    'diffuse_radiation': np.random.rand(1000),  # Replace with actual data
    'direct_radiation': np.random.rand(1000),  # Replace with actual data
    'high_cloud_cover': np.random.rand(1000)  # Replace with actual data
})
energy_data = np.random.rand(1000)  # Replace with actual energy production data

# 1. Preprocessing: Normalize weather data and energy data
scaler_weather = MinMaxScaler()
scaler_energy = MinMaxScaler()

# Normalize weather data (input features)
weather_data_scaled = scaler_weather.fit_transform(weather_data)

# Normalize energy data (target variable)
energy_data_scaled = scaler_energy.fit_transform(energy_data.reshape(-1, 1))

# Split into train and test sets (80% train, 20% test)
split_idx = int(0.8 * len(weather_data))
X_train, X_test = weather_data_scaled[:split_idx], weather_data_scaled[split_idx:]
y_train, y_test = energy_data_scaled[:split_idx], energy_data_scaled[split_idx:]

# 2. Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Only one output neuron for energy production
])

# 3. Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# 5. Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# 6. Inverse transform predictions to get the actual energy production values
predictions_scaled = model.predict(X_test)
predictions = scaler_energy.inverse_transform(predictions_scaled)

# 7. Visualizing the training progress
# Plot training & validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training & validation MAE
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Training MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

# 8. Visualizing Predictions vs True Values
plt.figure(figsize=(12, 6))
plt.plot(scaler_energy.inverse_transform(y_test.reshape(-1, 1)), label='True Energy Production')
plt.plot(predictions, label='Predicted Energy Production')
plt.title('Energy Production: True vs Predicted')
plt.xlabel('Samples')
plt.ylabel('Energy Production (kWh)')
plt.legend()
plt.show()
