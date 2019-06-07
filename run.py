import argparse
import logging
import time
import angle
import cv2
import os
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from os.path import join

logger = logging.getLogger('TfPoseEstimatorVideo')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    # 参数分析
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime video')
    parser.add_argument('--video', default=0, help="the source of video, default is camera")
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=432x368, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    # 模型初始化
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('video read+')

    # 捕捉视频
    video = cv2.VideoCapture(args.video)
    # 样本视频
    demo = cv2.VideoCapture(join('video', 'demo.mp4'))
    # 保存视频
    # fps = video.get(cv2.CAP_PROP_FPS)     #视频帧率
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    # frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # videoWriter = cv2.VideoWriter('demo.mp4', fourcc, fps, frame_size)

    # 读取视频
    ret_val1, image1 = video.read()
    ret_val2, image2 = demo.read()

    #视频读取完后退出
    while ret_val1 and ret_val2:

        # 识别骨骼
        humans = e.inference(image1, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            
        # logger.debug(humans)
        if len(humans) > 0:
            print(angle.CalAngle(humans[-1]))

        if not args.showBG:
            image1 = np.zeros(image1.shape)
        # 绘制骨骼
        image1 = TfPoseEstimator.draw_humans(image1, humans, imgcopy=False)
        # 保存视频
        # videoWriter.write(image)

        # 输出FPS
        cv2.putText(image1,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        # 显示实时视频与样本视频
        image = np.hstack((image1, image2))

        cv2.imshow('Human motion', image)
        fps_time = time.time()
        # 按下ESC时退出
        if cv2.waitKey(1) == 27:
            break

        # 读取视频
        ret_val1, image1 = video.read()
        ret_val2, image2 = demo.read()

    cv2.destroyAllWindows()

logger.debug('finished+')
