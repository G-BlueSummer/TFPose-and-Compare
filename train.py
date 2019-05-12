from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from os.path import join
from model import LSTM_Model
from data import DataSet
import time
import numpy as np

def train_models(train_data,train_labels,validation_data,validation_labels):
    batch_size = 128
    nb_epoch = 500

    callback = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ModelCheckpoint(
            filepath='LSTM.h5',
            monitor='val_loss',
            verbose=0,
            save_best_only=True
            )
        ]

    lstm = LSTM_Model(train_data.shape[1], train_data.shape[2])

    lstm.model.fit(
            train_data,
            train_labels,
            validation_data=(validation_data, validation_labels),
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=callback
            )
    
def main():
    data = DataSet()
    X_samples, y_samples = data.dataset_generator()

    expected_frames = 8
    labels = dict()

    print(X_samples[0].shape)
    print(len(X_samples))
    for i in range(len(X_samples)):
        X = X_samples[i]
        frames = X.shape[0]
        print(X.shape)
        if frames > expected_frames:
            X = X[0:expected_frames, :]
            X_samples[i] = X
        elif frames < expected_frames:
            temp = np.zeros(shape=(expected_frames, X.shape[1]))
            temp[0:frames, :] = X
            X_samples[i] = temp

    for y in y_samples:
        if y not in labels:
            labels[y] = len(labels)
    print(labels)
    for i in range(len(y_samples)):
        y_samples[i] = labels[y_samples[i]]

    nb_classes = len(labels)

    y_samples = np_utils.to_categorical(y_samples, nb_classes)

    print(y_samples)


if __name__ == '__main__':
    main()