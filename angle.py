from tf_pose import common
import numpy as np

#需要测量角度的部位，每个部位需要用它本身和与之连接的两个节点来计算角度
KeyPoints = [
    (1, 2, 3),      # 右肩
    (1, 5, 6),      # 左肩
    (2, 3, 4),      # 右臂
    (5, 6, 7),      # 左臂
    (1, 8, 9),      # 右胯
    (1, 11, 12),    # 左胯
    (8, 9, 10),     # 右膝
    (11, 12, 13),   # 左膝
]

#计算关键部位的夹角的余弦值
def CalAngle(human):
    #存放关键部位的角度
    angles = np.empty(len(KeyPoints))

    for i, keyPoint in enumerate(KeyPoints):
        #检测不全无法计算角度
        if keyPoint[0] not in human.body_parts.keys() or keyPoint[1] not in human.body_parts.keys() or keyPoint[2] not in human.body_parts.keys():
            angles[i] = np.nan
            continue

        #点
        part0 = human.body_parts[keyPoint[0]]
        part1 = human.body_parts[keyPoint[1]]
        part2 = human.body_parts[keyPoint[2]]

        #向量
        v1x = part0.x - part1.x
        v1y = part0.y - part1.y
        v2x = part2.x - part1.x
        v2y = part2.y - part1.y

        #夹角
        angles[i] = (v1x * v2x + v1y * v2y) / (np.sqrt(v1x * v1x + v1y * v1y) * np.sqrt(v2x * v2x + v2y * v2y))

    return angles

