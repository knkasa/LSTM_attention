# https://wire.insiderfinance.io/predicting-tomorrows-market-an-attention-based-lstm-model-for-stock-price-forecasting-1bad6ef4f643
# LSTM with Attention layer.

import yfinance as yf
import pandas as pd

# Fetch TSLA data
tsla_data = yf.download('TSLA', start='2021-01-01', end='2024-01-01')

# Remove the ticker symbol in column names if it exists
tsla_data.columns = tsla_data.columns.droplevel(1)


# Checking for missing values
tsla_data.isnull().sum()

# Filling missing values, if any
tsla_data.fillna(method='ffill', inplace=True)

# Drop NaN values created by rolling windows
tsla_data = tsla_data.dropna()

# Display the first few rows of the dataframe to verify
print(tsla_data.head())

from sklearn.preprocessing import MinMaxScaler

# Define the features to scale
features_to_scale = ['Adj Close']

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the selected features and transform them
tsla_data[features_to_scale] = scaler.fit_transform(tsla_data[features_to_scale])

# Display the first few rows of the scaled data to verify
print(tsla_data.head())

import numpy as np

X = []
y = []


# Convert the DataFrame to a NumPy array for easier indexing, focusing on the scaled features
data_values = tsla_data[features_to_scale].values

# Set the number of past days to use as features (e.g., 60 days)
sequence_length = 60

# Loop through the dataset to create sequences with the selected features
for i in range(sequence_length, len(data_values)):
    # Append the sequence of the past 'sequence_length' days for the selected features
    X.append(data_values[i-sequence_length:i])  
    
    # Append the target value, which is the 'Adj Close' price (use it as a target)
    y.append(data_values[i, 0])  
# Convert X and y to NumPy arrays
X, y = np.array(X), np.array(y)

# Display the shape of X and y to verify
print("Shape of X:", X.shape)  # Should be (number_of_samples, 60, 9)
print("Shape of y:", y.shape)  # Should be (number_of_samples,)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert lists to NumPy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

print("Final shape of X_train:", X_train.shape)
print("Final shape of y_train:", y_train.shape)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, Input
from keras.layers import BatchNormalization

model = Sequential()

# Specify input shape using an Input layer
model.add(Input(shape=(X_train.shape[1], 1)))

# Adding LSTM layers with return_sequences=True
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=True))

# Add any additional layers as needed
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=True))

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU
from keras.layers import AdditiveAttention, Multiply
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.regularizers import l2
from keras.optimizers import Adam

# Define function to create the model with attention
def create_model():
    lstm_units = 100  # LSTM units for capturing more patterns
    dropout_rate = 0.15  # Dropout rate
    l2_penalty = 0.01  # L2 regularization

    # Input layer
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))  # (60, 1)

    # First Bidirectional LSTM layer with attention
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_penalty)))(input_layer)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    # Attention mechanism
    attention = AdditiveAttention()([lstm_out, lstm_out])
    attention_output = Multiply()([lstm_out, attention])

    # Second Bidirectional LSTM layer
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(l2_penalty)))(attention_output)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    # Flatten and Dense layers for prediction
    flattened_output = Flatten()(lstm_out)
    flattened_output = Dense(64, activation=LeakyReLU(alpha=0.1))(flattened_output)
    output = Dense(1)(flattened_output)

    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

# Create the model and print the summary
model = create_model()
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
tensorboard = TensorBoard(log_dir='./logs')
csv_logger = CSVLogger('training_log.csv')
callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=callbacks_list)

# Plotting learning curves
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Val Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Convert X_test and y_test to Numpy arrays if they are not already
X_test = np.array(X_test)
y_test = np.array(y_test)

# Ensure X_test is reshaped similarly to X_train (i.e., with 1 features)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Now evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Making predictions
y_pred = model.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fetching the latest 6 months of TSLA stock data
data = yf.download('TSLA', period='6mo', interval='1d')

data.columns = data.columns.droplevel(1)

# Selecting the 'Adj Close' price and converting to numpy array
closing_prices = data['Adj Close'].values

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

# Since we need the last 60 days to predict the next day, we reshape the data accordingly
X_latest = scaled_data[-60:].reshape(1, 60, 1)  # Reshaping for LSTM input



# List to store predicted prices
predicted_stock_prices = []

# Making predictions for the next 3days
for i in range(3):
    # Make the prediction
    predicted_price_scaled = model.predict(X_latest)
    
    # Inverse scale the predicted price
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    predicted_stock_prices.append(predicted_price[0, 0])  # Save the predicted price
    
    # Prepare the new input for the next prediction
    new_input = np.concatenate((X_latest[0, 1:, :], predicted_price_scaled), axis=0)  # Update input with the predicted price
    X_latest = new_input.reshape(1, 60, 1)  # Reshape to maintain the input shape

# Print the predicted stock prices for the next 3days
print("Predicted Adjusted Close Prices for the next 3days: ", predicted_stock_prices)

import yfinance as yf
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# Fetching the latest 6 months of TSLA stock data
data = yf.download('TSLA', period='6mo', interval='1d')

# Dropping unnecessary columns and setting up the data
data.columns = data.columns.droplevel(1)

# Assuming `predicted_stock_prices` contains the predictions for the next 3 days
predicted_stock_prices = np.array(predicted_stock_prices).reshape(-1, 1)

# Create a list of dates for the predictions
last_date = data.index[-1]
prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=3)

# Create a DataFrame for predictions
predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_stock_prices, columns=['Adj Close'])

# Define a custom style for better visualization
mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='skyblue')
s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', figcolor='white', gridcolor='lightgray')

# Plot the actual data with mplfinance and overlay predictions
mpf.plot(data, type='line', style=s, volume=True, title="TESLA Stock Price with Predicted Next 3 Days")

# Overlay the predicted prices as a separate plot
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Adj Close'], color='blue', label='Historical Data')
plt.plot(predictions_df.index, predictions_df['Adj Close'], linestyle='--', marker='o', color='orange', markersize=8, label='Predicted Prices')

# Add titles, labels, and legend
plt.title("TESLA Stock Price with Predicted Next 3 Days")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price (USD)")
plt.legend(loc='upper left')

# Display the plot
plt.show()
