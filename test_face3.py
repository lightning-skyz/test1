"""
代码功能：
1. 用dlib人脸检测器检测出人脸，返回的人脸矩形框
2. 对检测出的人脸进行关键点检测并用圈进行标记
3. 将检测出的人脸关键点信息写到txt文本中
"""
import cv2
import dlib
import numpy as np


def face(filePath):
    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
    predictor = dlib.shape_predictor(predictor_model)

    # cv2读取图像
    output_pos_info = "F:/output/Messi.txt"
    img = cv2.imread(filePath)
    file_handle = open(output_pos_info, 'a')
    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 人脸数rects（rectangles）
    rects = detector(img_gray, 0)

    # 如果检测到人脸
    if len(rects) != 0:
        for i in range(len(rects)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
            for idx, point in enumerate(landmarks):
                # # 68点的坐标
                pos = (point[0, 0], point[0, 1])
                # print(idx+1, pos)
                pos_info = str(point[0, 0]) + ' ' + str(point[0, 1]) + '\n'
                file_handle.write(pos_info)
                # 利用cv2.circle给每个特征点画一个圈，共68个
                cv2.circle(img, pos, 3, color=(0, 255, 0))
                # 利用cv2.putText输出1-68

                # cv2.putText(img, str(idx+1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(img, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    file_handle.close()
    cv2.imwrite("F:/output/4_keypoints.png", img)

