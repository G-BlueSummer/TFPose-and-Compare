from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from os.path import join
from model import LSTM_Model
from data import DataSet
import time
import numpy as np

def train_model(train_data,train_labels,validation_data,validation_labels, nb_classes):
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

    lstm = LSTM_Model(train_data.shape[1], train_data.shape[2], nb_classes)

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

    return lstm.model
    
def main():
    # data = DataSet()  #载入数据
    X_samples = np.load(join('features', 'X_samples.npy'))
    y_samples = np.load(join('features', 'y_samples.npy'))
    X_train, X_validate, y_train, y_validate = train_test_split(X_samples, y_samples)

    print(y_validate)

    nb_classes = y_samples.shape[1]
    # train_model(X_train, X_validate, y_train, y_validate, nb_classes)


if __name__ == '__main__':
    main()