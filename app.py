
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

# Your OpenWeatherMap API key
API_KEY = "YOUR_API_KEY"
CITY = "Bangalore"  # change city as needed

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    # Check for successful API call
    if response.get("cod") != 200:
        print(f"Error fetching weather data: {response.get('message', 'Unknown error')}")
        return None

    try:
        # Extract required fields
        temp = response['main']['temp'] / 41  # normalize like dataset (max ~41C)
        atemp = temp  # approximation (dataset used 'feels like')
        hum = response['main']['humidity'] / 100
        windspeed = response['wind']['speed'] / 67  # dataset max ~67
        weather_id = response['weather'][0]['id']

        # Map OpenWeatherMap weather_id → dataset's weathersit
        if weather_id < 600:
            weathersit = 3  # Rain
        elif weather_id < 700:
            weathersit = 2  # Mist
        elif weather_id < 800:
            weathersit = 3
        elif weather_id == 800:
            weathersit = 1  # Clear
        else:
            weathersit = 2  # Clouds

        return {
            "holiday": 0,
            "workingday": 1,
            "weathersit": weathersit,
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed
        }
    except KeyError as e:
        print(f"Error processing weather data: Missing key {e}")
        return None


# Ask user
date_input = input("Enter date to predict bike rentals (YYYY-MM-DD): ")
date_obj = datetime.strptime(date_input, "%Y-%m-%d")

weather_data = get_weather(CITY)  # fetch live weather

if weather_data:
    mnth = date_obj.month
    weekday = date_obj.weekday()
    yr = date_obj.year - 2011
    day = date_obj.day
    year = date_obj.year

    # Season mapping
    if mnth in [12, 1, 2]:
        season = 4
    elif mnth in [3, 4, 5]:
        season = 1
    elif mnth in [6, 7, 8]:
        season = 2
    else:
        season = 3

    # Prepare input (as DataFrame to keep feature names)
    input_data = [[season, yr, mnth, weather_data["holiday"], weekday,
                   weather_data["workingday"], weather_data["weathersit"],
                   weather_data["temp"], weather_data["atemp"], weather_data["hum"],
                   weather_data["windspeed"], year, mnth, day]]

    input_df = pd.DataFrame(input_data, columns=X.columns)

    # Scale and predict
    input_scaled = scaler_X.transform(input_df)
    predicted_scaled = model.predict(input_scaled)
    predicted_count = scaler_y.inverse_transform(predicted_scaled)

    print(f"\nPredicted bike rentals for {CITY} on {date_input}: {int(predicted_count[0][0])}")

else:
    print("Could not predict bike rentals due to weather data fetching error.")
