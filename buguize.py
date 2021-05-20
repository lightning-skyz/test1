"""
代码功能：
1. 用dlib人脸检测器检测出人脸，返回的人脸矩形框
2. 分割出人脸并保存
"""
import numpy as np
import cv2
import dlib



def buguize(filePath):
    def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
        # get the mask of the image
        if image_landmarks.shape[0] != 68:
            raise Exception(
                'get_image_hull_mask works only with 68 landmarks')
        int_lmrks = np.array(image_landmarks, dtype=np.int)

        hull_mask = np.zeros(image_shape[0:2] + (1,), dtype=np.float32)

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[0:9],
                            int_lmrks[17:18]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[8:17],
                            int_lmrks[26:27]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:20],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[24:27],
                            int_lmrks[8:9]))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[19:25],
                            int_lmrks[8:9],
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[17:22],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((int_lmrks[22:27],
                            int_lmrks[27:28],
                            int_lmrks[31:36],
                            int_lmrks[8:9]
                            ))), (1,))

        # nose
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

        if ie_polys is not None:
            ie_polys.overlay_mask(hull_mask)

        return hull_mask

    def merge(img_1, mask):
        # merge rgb and mask into a rgba image
        r_channel, g_channel, b_channel = cv2.split(img_1)
        if mask is not None:
            alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
            alpha_channel *= mask * 255
        else:
            alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        return img_BGRA

    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
    predictor = dlib.shape_predictor(predictor_model)

    image = cv2.imread(filePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 取灰度

    # 人脸数rects（rectangles）
    rects = detector(image, 0)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])

    # print(landmark)
    mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)
    # cv2.imshow("mask", (mask*255).astype(np.uint8))

    image_bgra = merge(image, mask)
    # cv2.imshow("image_bgra", image_bgra)
    # cv2.waitKey(1)
    cv2.imwrite('F:/output/xiaozu_buguize.png', image_bgra)



