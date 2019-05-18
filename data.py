import numpy as np
import os
import cv2
import logging
from os.path import join
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from keras.utils import np_utils

class DataSet():
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(relativeCreated)6d %(threadName)s %(message)s')
        self.max_frames = 8     #设置固定帧数为一个样本
        self.resize = '432x368'
        self.model = 'cmu'
        self.estimator = self.load_tf_pose()

    #加载骨架识别的模型
    def load_tf_pose(self):
        w, h = model_wh(self.resize)
        return TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

    #生成一个视频文件的特征
    def extract_video_features(self, video_input_file_path):
        video = cv2.VideoCapture(video_input_file_path)
        logging.info("Video load: {}".format(video_input_file_path))
        features = []
        while video.isOpened():
            ret_val, image = video.read()
            # 视频读取结束
            if not ret_val:
                break

            feature = TfPoseEstimator.trans_data(self.estimator.inference(image, upsample_size=5.0))     #只提取第一个人
            if(len(feature) > 1):
                continue
            features.append(feature[0])
        features = np.array(features)
        logging.info("Features shape: {}".format(features.shape))
        return features

    #生成数据集
    def dataset_generator(self):
        X_samples = []
        y_samples = []
        
        for c in os.listdir('video'):
            for file in os.listdir(join('video', c)):
                video_path = join('video', c, file)
                X = self.extract_video_features(video_path)
                y = c
                X_samples.append(X)
                y_samples.append(y)

        X_samples, y_samples = self.trans_data(X_samples, y_samples)

        #保存到文件方便下次存取
        np.save(join('features', 'X_samples.npy'), X_samples)
        np.save(join('features', 'y_samples.npy'), y_samples)

        return X_samples, y_samples

    #将数据转换为可用的样本数据
    def trans_data(self, X_samples, y_samples):
        #按帧切取
        X = []
        y = []
        for i, X_sample in enumerate(X_samples):
            start = 0
            stop = self.max_frames
            while stop < len(X_sample):
                X.append(X_sample[start:stop])
                y.append(y_samples[i])
                start = stop
                stop += self.max_frames 

        X_samples = np.array(X)
        y_samples = np.array(y)

        logging.info("X_samples shape: {}".format(X_samples.shape))
        logging.info("y_samples shape: {}".format(y_samples.shape))

        labels = dict()
        #给类别编号并放入字典
        for y in y_samples:
            if y not in labels:
                labels[y] = len(labels)
        for i in range(len(y_samples)):
            y_samples[i] = labels[y_samples[i]]

        nb_classes = len(labels)
        #类似于独热编码
        y_samples = np_utils.to_categorical(y_samples, nb_classes)

        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)

        return X_samples, y_samples

    @staticmethod
    def load_data():
        X_samples = np.load(join('features', 'X_samples.npy'))
        y_samples = np.load(join('features', 'y_samples.npy'))
        return X_samples, y_samples

if __name__ == '__main__':
    data = DataSet()
    data.dataset_generator()