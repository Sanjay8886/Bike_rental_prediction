
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import json

import requests, zipfile, io

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
df = pd.read_csv(z.open('day.csv'))

# Keep date for plotting and user inputs
df['dteday'] = pd.to_datetime(df['dteday'])

# Drop unnecessary columns
df.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)

# Extract year, month, day from dteday
df['year'] = df['dteday'].dt.year
df['month'] = df['dteday'].dt.month
df['day'] = df['dteday'].dt.day
df['weekday'] = df['dteday'].dt.weekday

X = df.drop(['cnt', 'dteday'], axis=1)
y = df['cnt']

# Scale features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

# Split into train/test
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X_scaled, y_scaled, df['dteday'], test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=200, batch_size=32,
                    callbacks=[early_stop], verbose=1)

y_pred = model.predict(X_test)
y_test_orig = scaler_y.inverse_transform(y_test)
y_pred_orig = scaler_y.inverse_transform(y_pred)

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2 = r2_score(y_test_orig, y_pred_orig)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Sort test data by date
sorted_idx = np.argsort(dates_test)
dates_sorted = dates_test.iloc[sorted_idx]
y_test_sorted = y_test_orig[sorted_idx]
y_pred_sorted = y_pred_orig[sorted_idx]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates_sorted,
    y=y_test_sorted.flatten(),
    mode='lines+markers',
    name='Actual Count',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Actual: %{y}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=dates_sorted,
    y=y_pred_sorted.flatten(),
    mode='lines+markers',
    name='Predicted Count',
    line=dict(color='red', dash='dash'),
    hovertemplate='Date: %{x}<br>Predicted: %{y}<extra></extra>'
))
fig.update_layout(
    title="Actual vs Predicted Bike Rentals",
    xaxis_title="Date",
    yaxis_title="Bike Rentals",
    hovermode="x unified",
    template="plotly_white"
)
fig.show()

# 4️⃣ Plot Training vs Validation Loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.show()

weather_json = {
    "2025-08-22": {"holiday":0,"workingday":1,"weathersit":2,"temp":0.6,"atemp":0.62,"hum":0.45,"windspeed":0.1},
    "2025-08-23": {"holiday":0,"workingday":1,"weathersit":1,"temp":0.5,"atemp":0.52,"hum":0.4,"windspeed":0.1},
    # Add more dates as needed
}

# User input
date_input = input("Enter date to predict bike rentals (YYYY-MM-DD): ")
try:
    date_obj = datetime.strptime(date_input, "%Y-%m-%d")
except:
    print("Invalid date format!")
    exit()

# Extract features
mnth = date_obj.month
weekday = date_obj.weekday()
yr = date_obj.year - 2011
day = date_obj.day
year = date_obj.year

# Season mapping
if mnth in [12,1,2]:
    season=4
elif mnth in [3,4,5]:
    season=1
elif mnth in [6,7,8]:
    season=2
else:
    season=3

# Fetch weather
if date_input in weather_json:
    w = weather_json[date_input]
else:
    print("Weather data not available for this date!")
    exit()

# Prepare input array
input_data = np.array([[season, yr, mnth, w["holiday"], weekday,
                        w["workingday"], w["weathersit"],
                        w["temp"], w["atemp"], w["hum"], w["windspeed"],
                        year, mnth, day]])

# Scale and predict
input_scaled = scaler_X.transform(input_data)
predicted_scaled = model.predict(input_scaled)
predicted_count = scaler_y.inverse_transform(predicted_scaled)

print(f"\nPredicted bike rentals for {date_input}: {int(predicted_count[0][0])}")
