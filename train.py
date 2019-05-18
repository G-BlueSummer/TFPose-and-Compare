from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from os.path import join
from model import LSTM_Model
from data import DataSet
import time
import numpy as np

def train_model(train_data,train_labels,validation_data,validation_labels, nb_classes):
    batch_size = 128
    nb_epoch = 1000
    timestamp = time.time()
    callback = [
        EarlyStopping(patience=10),
        ModelCheckpoint(
            filepath=join('models', 'LSTM-{}.hdf5'.format(timestamp)),
            verbose=1,
            save_best_only=True
            ),
        CSVLogger(join('logs', 'LSTM-{}.log'.format(timestamp)))    #记录训练情况，用时间命名
        ]

    lstm = LSTM_Model(train_data.shape[1], train_data.shape[2], nb_classes)

    lstm.model.fit(
            train_data,
            train_labels,
            validation_data=(validation_data, validation_labels),
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            callbacks=callback
            )

    return lstm.model
    
def main():
    X_samples, y_samples = DataSet.load_data()      #载入数据
    X_train, X_validate, y_train, y_validate = train_test_split(X_samples, y_samples, test_size=0.2, random_state=0)

    nb_classes = y_samples.shape[1]
    train_model(X_train, y_train, X_validate, y_validate, nb_classes)


if __name__ == '__main__':
    main()