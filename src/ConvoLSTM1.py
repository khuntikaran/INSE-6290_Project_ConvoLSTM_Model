import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from tensorflow.python.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def load_data(filepath):
    return pd.read_csv(filepath)


def apply_word2vec(textual_data):
    
    model = Word2Vec(sentences=textual_data, vector_size=100, window=5, min_count=1, workers=4)
   
    return model

def preprocess_data(df):
    df = df.drop(['Year', 'Month', 'Week'], axis=1, errors='ignore')
    X = df.drop('Weekly_Sales', axis=1)
    y = df['Weekly_Sales']
    
   
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Reshape data for ConvLSTM
def reshape_for_convlstm(X, time_steps, rows, cols):
    return X.reshape(-1, time_steps, rows, cols, 1)

# Create a ConvLSTM neural network
def create_convlstm_model(input_shape):
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def correlation_clustering(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()



if __name__ == "__main__":
    
    df = load_data('data/Processed/feature_engineered_retail_sales_data.csv')
    

    X, y, scaler = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape the input data for ConvLSTM
    time_steps = 1
    rows, cols = 1, X_train.shape[1]
    X_train = reshape_for_convlstm(X_train, time_steps, rows, cols)
    X_test = reshape_for_convlstm(X_test, time_steps, rows, cols)
    
    # Create the ConvLSTM neural network
    convlstm_model = create_convlstm_model(X_train.shape[1:])
    
    # Train the ConvLSTM neural network
    convlstm_model.fit(X_train, y_train, epochs=60, batch_size=64, verbose=2)
    

    # Evaluate the ConvLSTM neural network
    y_pred = convlstm_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    relative_error_percentage = np.mean(np.abs((y_test - y_pred.squeeze()) / y_test)) * 100
    r_squared = r2_score(y_test, y_pred)
   
    



    relative_error_percentage = np.mean(np.abs((y_test - y_pred.squeeze()) / y_test)) * 100
    print(f"R-Squared: {r_squared:.2f}")
    print(f"Model Performance on Test Set - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Relative Error (%): {relative_error_percentage:.2f}")
    print(f"MSE: {mse:.2f}")
  
    correlation_clustering(df)
    import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Sales')
    plt.plot(y_pred, label='Predicted Sales', alpha=0.7)
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Data Points')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

def plot(y_test, y_pred):
   plt.figure(figsize=(8, 6))
   plt.scatter(y_test, y_pred, alpha=0.5)
   plt.xlabel('Actual Sales')
   plt.ylabel('Predicted Sales')
   plt.title('Actual vs Predicted Sales')
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) 
   plt.show()

plot_predictions(y_test, y_pred)
plot(y_test, y_pred)

    
