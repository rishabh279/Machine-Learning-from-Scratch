from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from utils.utils import one_hot_conversion
from keras.models import Sequential

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('../data/fashion-mnist_train.csv')
data = data.values

x = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
y = data[:, 0].astype(np.int32)

k = len(set(y))
y = one_hot_conversion(y)


model = Sequential()

model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=128, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=k))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
print(model.summary())
r = model.fit(x, y, validation_split=0.1, epochs=15, batch_size=32)
print("Returned:", r)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

