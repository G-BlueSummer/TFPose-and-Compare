import numpy as np
import os
import cv2
import logging
from os.path import join
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from angle import CalAngle

class DataSet():
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(relativeCreated)6d %(threadName)s %(message)s')
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

            feature = self.estimator.inference(image, upsample_size=5.0)[-1]     #只提取第一个人
            features.append(CalAngle(feature))

        features = np.array(features)
        print(features)
        logging.info("Features shape: {}".format(features.shape))
        return features

    #生成数据集
    def dataset_generator(self):
        for video_path in os.listdir('video'):
            X = self.extract_video_features(join('video', video_path))

            #保存到CSV
            np.savetxt(join('features', video_path + '.csv'), np.array(X), delimiter=',')


if __name__ == '__main__':
    data = DataSet()
    data.dataset_generator()