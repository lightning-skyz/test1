"""
代码功能：
  对检测出的人脸进行关键点检测并根据关键点数组切割出人脸的五官：嘴巴，内嘴唇，左右眉毛，左右眼，鼻子，下巴
"""
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt

def fenge(filePath):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(filePath)
    image = imutils.resize(image, width=500)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    rects = detector(img_gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(img_gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i: j]]))
                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

                # plt.savefig("F:/output/{}.png".format(name))
                # plt.clf()

            cv2.imshow("ROI", roi)
            cv2.imshow("image", clone)
            cv2.waitKey(0)



