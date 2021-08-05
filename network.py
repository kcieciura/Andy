
import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras.layers as tfl
import dsloader
import keras
from sklearn.model_selection import train_test_split

batch_size = 16
epochs = 5

X, y = dsloader.loadDataset()
y = tf.one_hot(y, dsloader.NUM_CLASSES)  # One hot encoding of y, also converts from numpy ndarry to eager tensor
y = np.array(y)  # convert back to np array or else train_test_split doesnt work
X = X.reshape(X.shape[0], dsloader.IMG_ROWS, dsloader.IMG_COLS, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)


def convNN():
    model=tf.keras.Sequential([

        tfl.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 1)),
        tfl.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tfl.MaxPool2D(),
        tfl.Dropout(0.25),

        tfl.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        tfl.Conv2D(filters=128, kernel_size=3, activation='relu'),
        tfl.MaxPool2D(),
        tfl.Dropout(0.25),
        tfl.Flatten(),

        tfl.Dense(units=1024, activation='relu'),
        tfl.Dropout(0.5),
        tfl.Dense(units=dsloader.NUM_CLASSES, activation='softmax')
    ])

    return model


#model = convNN()

#model.compile(loss='categorical_crossentropy',
            # optimizer='adam',
            # metrics=['accuracy'])

#model.fit(X_train, y_train,
         # batch_size=batch_size,
         # epochs=epochs)
#model.save('andy_model')

model = keras.models.load_model('andy_model')

model.evaluate(X_test, y_test)
