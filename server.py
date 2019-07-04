from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import angle

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from sklearn.metrics import r2_score
from os.path import join

app = Flask(__name__)
CORS(app)

# 设置参数
model = 'cmu'
resize = '432x368'
upsample_size = 4.0
    
# 模型初始化
w, h = model_wh(resize)
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

# 样本数据
stdAngle = np.loadtxt(join('features', 'demo.tsv'), delimiter='\t')

@app.route('/', methods=['POST'])
def compare():
    data = request.form['snapData']                     # 读取图片并去掉头部信息
    data = data.split(',')[1]                           # 去掉头部信息
    frame = int(float(request.form['frame']) * 30)      # 当前帧时间

    print(frame)
    
    image = np.fromstring(base64.urlsafe_b64decode(data), np.uint8)      # 解码
    image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)

    if image is None:
        return jsonify({'code': 0, 'msg': 'Image can not be read'})

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=upsample_size)

    # 检测到人
    if len(humans) > 0:
        realAngle = angle.CalAngle(humans[-1])
        if np.isnan(realAngle).any():
            return jsonify({'code': 2, 'msg': '识别部位缺失'})

        i = np.abs(realAngle - stdAngle[frame]) < 0.15        # 设置允许误差
        realAngle[i] = stdAngle[frame][i]
        score = np.sqrt(r2_score(stdAngle[frame], realAngle))
        # print(score)
        return jsonify({'code': 3, 'msg': '识别成功', 'score': score})

    return jsonify({'code': 1, 'msg': '无法检测到人'})

if __name__ == '__main__':
    app.run(debug=True) 
