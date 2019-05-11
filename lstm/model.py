from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam


class LSTM_Model():
    def __init__(self, n_classes, seq_length, features_length=2048):
        self.model = self.lstm()
        self.input_shape = (seq_length, features_length)    #输入的特征形状
        self.n_classes = n_classes                          #分类的类别数

        metrics = ['accuracy']
        if self.n_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

        pass

    def lstm(self):
        model = Sequential()
        model.add(
            LSTM(2048, return_sreturn_sequences=False,
            input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    