import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.drop(['Year', 'Month', 'Week'], axis=1, errors='ignore')
    X = df.drop('Weekly_Sales', axis=1)
    y = df['Weekly_Sales']
    
    # Scale the features to a range of [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def create_complex_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5), 
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5), 
        keras.layers.LSTM(128),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),  
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),  
        keras.layers.Dense(1)  
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
 
    df = load_data('Data/Processed/feature_engineered_retail_sales_data.csv')
    
    
    X, y, scaler = preprocess_data(df)
    
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
   
    timesteps = 1 
    input_shape = (timesteps, X_train.shape[1])
    X_train = X_train.reshape(-1, timesteps, X_train.shape[1])
    X_test = X_test.reshape(-1, timesteps, X_test.shape[1])
    
    complex_lstm_model = create_complex_lstm_model(input_shape)
    
    complex_lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2)
    
    
    y_pred = complex_lstm_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)  
    y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)) 
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model Performance on Test Set - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    complex_lstm_model.save('models/complex_lstm_model.h5')
