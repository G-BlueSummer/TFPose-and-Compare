import numpy as np
import threading
import os
from os.path import join

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


class DataSet():
    def __init__(self, seq_length=40):
        self.max_frames = 300
        self.seq_length = seq_length
        self.classes = self.get_classes()
        self.data = self.get_data()

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test):
        while True:
            X, y = [], []

            sequence = None

            X.append(sequence)
            
            yield np.array(X), np.array(y)
            
    def get_classes(self):
        classes = []
        # 类别名按目录放置
        for dir in os.listdir('video'):
            classes.append(dir)

        return classes

    def get_data(self):
        data = []

        for dir in self.get_classes():
            for video in os.listdir(join('video', dir)):
                data.append((join('video', dir, video), dir))

        return data