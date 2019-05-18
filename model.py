from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, SGD


class LSTM_Model():
    def __init__(self, seq_length, features_length, n_classes):
        self.input_shape = (seq_length, features_length)    #输入的特征形状
        self.n_classes = n_classes                          #分类的类别数
        self.model = self.lstm()

        metrics = ['accuracy']
        if self.n_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lstm(self):
        model = Sequential()
        model.add(LSTM(1024, input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model