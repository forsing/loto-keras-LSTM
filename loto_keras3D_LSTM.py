"""        
=== System Information ===
Python version                 3.11.13        
macOS Apple                    Tahos 
Apple                          M1
"""



"""
bash

pip uninstall -y keras keras-core keras-tuner tensorflow tensorflow-macos tensorflow-metal
pip install tensorflow-macos keras
python3 -c "from keras.models import Sequential; print('✅ Radi!')"
"""



import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# -----------------------------
# 1️⃣ Deterministički seed
# -----------------------------
SEED = 39
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# 2️⃣ Učitavanje i normalizacija
# -----------------------------
df = pd.read_csv('/data/loto7_4506_k87.csv', header=None)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# -----------------------------
# 3️⃣ Kreiranje sekvenci
# -----------------------------
look_back = 5

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[0:train_size,:], scaled_data[train_size:,:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Input shape for LSTM: [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, 7))
testX = np.reshape(testX, (testX.shape[0], look_back, 7))

# -----------------------------
# 4️⃣ LSTM model
# -----------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, 7), activation='tanh', return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='linear'))  # linear za regresiju brojeva

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# -----------------------------
# 5️⃣ Fit
# -----------------------------
model.fit(trainX, trainY, epochs=300, batch_size=50, verbose=1)




# ---------------------------------
# 6️⃣ Predikcija sledećeg izvlačenja
# ---------------------------------
# uzmi poslednju sekvencu iz testX
last_seq = testX[-1].reshape(1, look_back, 7)  # pravilno reshapen u 3D

pred_scaled = model.predict(last_seq)
pred = scaler.inverse_transform(pred_scaled)

next_numbers = np.round(pred).astype(int)[0]
print()
print(f'The predicted next set of numbers is: {next_numbers}')
print()
"""
The predicted next set of numbers is: [ 3  7 x x x 33 37]
"""




 
