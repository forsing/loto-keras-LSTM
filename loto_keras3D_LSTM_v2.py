
"""
bash

pip uninstall -y keras keras-core keras-tuner tensorflow tensorflow-macos tensorflow-metal
pip install tensorflow-macos keras
python3 -c "from keras.models import Sequential; print('✅ Radi!')"
"""


"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
svih 4584 izvlacenja
30.07.1985.- 20.03.2026.
"""


import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
df = pd.read_csv('/Users/4c/Desktop/GHQ/data/loto7_4584_k23.csv', header=None)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# -----------------------------
# 3️⃣ Kreiranje sekvenci
# -----------------------------
look_back = 9

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
model.add(LSTM(128, input_shape=(look_back, 7), activation='tanh', return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(64, activation='tanh', return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='linear'))  # linear za regresiju brojeva

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))

# -----------------------------
# 5️⃣ Fit
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, min_delta=1e-7),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=5e-7, verbose=1),
]

model.fit(trainX, trainY, epochs=900, batch_size=24, verbose=1, validation_split=0.15, callbacks=callbacks)




# ---------------------------------
# 6️⃣ Predikcija sledećeg izvlačenja
# ---------------------------------
# uzmi poslednju sekvencu iz testX
last_seq = testX[-1].reshape(1, look_back, 7)  # pravilno reshapen u 3D

pred_scaled = model.predict(last_seq, verbose=0)
pred = scaler.inverse_transform(pred_scaled)

next_numbers = np.clip(np.round(pred).astype(int)[0], 1, 39)
print()
print(f'The predicted next set of numbers is: {next_numbers}')
print()

# dodatni ispis za brzu proveru kvaliteta
test_loss = model.evaluate(testX, testY, verbose=0)
print(f'Test MSE: {test_loss:.8f}')

# čuvanje v2 modela
model.save('loto_keras3D_LSTM_v2.keras')
print('Saved model: loto_keras3D_LSTM_v2.keras')


"""
124/130 [===========================>..] - ETA: 0s - loss: 0.0130/130 [==============================] - 1s 7ms/step - loss: 0.0305 - val_loss: 0.0288 - lr: 1.5625e-05

The predicted next set of numbers is: 
[ 5 10 15 20 25 30 35]

Test MSE: 0.03093841
"""





"""
python 3.11.13

keras-3.12.0 
 
tensorflow-macos-2.16.2

tensorboard-2.16.2 
tensorflow-2.16.2 
tensorflow-io-gcs-filesystem-0.37.1 
ml-dtypes-0.3.2
typing-extensions-4.15.0

numpy 1.26.4
"""


 

