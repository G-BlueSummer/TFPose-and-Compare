import numpy as np
import os
import cv2
from os.path import join
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from keras.utils import np_utils

class DataSet():
    def __init__(self):
        self.max_frames = 8
        self.resize = '432x368'
        self.model = 'mobilenet_thin'
        self.estimator = self.load_tf_pose()

    #加载骨架识别的模型
    def load_tf_pose(self):
        w, h = model_wh(self.resize)
        return TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

    #生成一个视频文件的特征
    def extract_video_features(self, video_input_file_path):
        video = cv2.VideoCapture(video_input_file_path)
        features = []
        while video.isOpened():
            ret_val, image = video.read()
            # 视频读取结束
            if not ret_val:
                break

            feature = TfPoseEstimator.trans_data(self.estimator.inference(image, upsample_size=5.0))[0]     #只提取第一个人
            features.append(feature)

        return np.array(features)

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
        expected_frames = 8     #设置固定帧数为一个样本

        for i in range(len(X_samples)):
            X = X_samples[i]
            frames = X.shape[0]
            if frames > expected_frames:
                X = X[0:expected_frames, :]
                X_samples[i] = X
            elif frames < expected_frames:
                temp = np.zeros(shape=(expected_frames, X.shape[1]))
                temp[0:frames, :] = X
                X_samples[i] = temp

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


if __name__ == '__main__':
    data = DataSet()
    data.dataset_generator()