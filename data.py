import numpy as np
import os
import cv2
from os.path import join
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

class DataSet():
    def __init__(self):
        self.max_frames = 300
        self.resize = '432x368'
        self.model = 'cmu'
        self.classes = self.get_classes()
        self.data = self.get_data()
        self.estimator = self.load_tf_pose()

    #加载骨架识别的模型
    def load_tf_pose(self):
        w, h = model_wh(self.resize)
        return TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))


    #生成每一帧的特征
    def frame_generator(self):
        #每个data是个视频文件名与对应的类别
        for name, cla in self.data:
            video = cv2.VideoCapture(name)
            while video.isOpened():
                ret_val, image = video.read()
                # 视频读取结束
                if not ret_val:
                    break

                _, X  = self.estimator.inference(image, upsample_size=4.0)
                y = cla

                yield np.array(X), np.array(y)
            
    #生成类别
    def get_classes(self):
        classes = []
        # 类别名按目录放置
        for dir in os.listdir('video'):
            classes.append(dir)

        return classes

    #生成视频和与之对应的标签
    def get_data(self):
        data = []

        for dir in self.classes:
            for video in os.listdir(join('video', dir)):
                data.append((join('video', dir, video), dir))

        return data

if __name__ == '__main__':
    data = DataSet()
    frame = data.frame_generator()
    while True:
        print(next(frame))