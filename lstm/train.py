from keras.callbacks import ModelCheckpoint, EarlyStopping
from os.path import join
from lstm.model import LSTM_Model
from lstm.data import DataSet
import time

def train_models(train_data,train_labels,validation_data,validation_labels):
    # 模型保存

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
    pass

if __name__ == '__main__':
    main()