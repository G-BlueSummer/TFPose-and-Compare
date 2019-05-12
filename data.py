import numpy as np
import os
import cv2
from os.path import join
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

class DataSet():
    def __init__(self):
        self.max_frames = 8
        self.resize = '432x368'
        self.model = 'cmu'
        self.estimator = self.load_tf_pose()

    #加载骨架识别的模型
    def load_tf_pose(self):
        w, h = model_wh(self.resize)
        return TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

    #生成一个视频文件的特征
    def extract_video_features(self, video_input_file_path, feature_output_file_path=None):
        video = cv2.VideoCapture(video_input_file_path)
        features= []
        while video.isOpened():
            ret_val, image = video.read()
            # 视频读取结束
            if not ret_val:
                break

            feature = TfPoseEstimator.trans_data(self.estimator.inference(image, upsample_size=5.0))
            features.append(feature)
        unscaled_features = np.array(features)
        if feature_output_file_path:
            np.save(feature_output_file_path, unscaled_features)    #保存到文件
        return unscaled_features

    #生成每一帧的特征
    # def frame_generator(self):
    #     #每个data是个视频文件名与对应的类别
    #     for name, label in self.data:
    #         video = cv2.VideoCapture(name)
    #         while video.isOpened():
    #             ret_val, image = video.read()
    #             # 视频读取结束
    #             if not ret_val:
    #                 break

    #             X  = TfPoseEstimator.trans_data(self.estimator.inference(image, upsample_size=5.0))

    #             yield X, label

    #生成数据集
    def dataset_generator(self, save_data=True):
        X_samples = []
        y_samples = []

        output_feature_file_path = None
        
        for c in os.listdir('video'):
            for file in os.listdir(join('video', c)):
                video_path = join('video', c, file)
                if save_data:
                    output_feature_file_path = join('features' ,file.split('.')[-2] + '.npy')
                X = self.extract_video_features(video_path, output_feature_file_path)
                y = c
                X_samples.append(X)
                y_samples.append(y)

        return X_samples, y_samples


if __name__ == '__main__':
    data = DataSet()
    data.dataset_generator()