from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from os.path import join
from lstm.model import LSTM_Model
from lstm.data import DataSet
import time

def train(seq_length, batch_size=32, nb_epoch=100):
    # 模型保存
    checkPointer = ModelCheckpoint(
        filepath='model.hdf5',
        verbose=1,
        save_best_only=True
    )

    tb = TensorBoard(log_dir=join('logs, model'))

    early_stopper = EarlyStopping(patience=5)

    timestamp = time.time()
    csv_logger = CSVLogger(join('logs', str(timestamp) + '.log'))

    #导入数据集
    data = DataSet(seq_length)

    steps_per_epoch = (len(data.data) * 0.7)

    #导入生成器
    generator = data.frame_generator(batch_size, 'train')
    val_generator = data.frame_generator(batch_size, 'test')

    lstm = LSTM_Model(len(data.classes), seq_length)
    #使用生成器训练
    lstm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkPointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)
    

def main():
    seq_length = 40
    train(seq_length)

if __name__ == '__main__':
    main()