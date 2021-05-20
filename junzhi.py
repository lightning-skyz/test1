"""
代码功能：
1. 添加椒盐噪声和高斯噪声
2. 使用均值滤波、中值滤波以及cv2.fastNlMeansDenoisingColored函数去噪
"""
import numpy as np

import random

import cv2

from matplotlib import pyplot as plt

def junzhi(filePath):
    def sp_noise(image, prob):
        """

        添加椒盐噪声

        prob:噪声比例

        """

        output = np.zeros(image.shape, np.uint8)

        thres = 1 - prob

        for i in range(image.shape[0]):

            for j in range(image.shape[1]):

                rdn = random.random()

                if rdn < prob:

                    output[i][j] = 0

                elif rdn > thres:

                    output[i][j] = 255

                else:

                    output[i][j] = image[i][j]

        return output

    def gasuss_noise(image, mean=0, var=0.001):
        """

            添加高斯噪声

            mean : 均值

            var : 方差

        """

        image = np.array(image / 255, dtype=float)

        noise = np.random.normal(mean, var ** 0.5, image.shape)

        out = image + noise

        if out.min() < 0:

            low_clip = -1.

        else:

            low_clip = 0.

        out = np.clip(out, low_clip, 1.0)

        out = np.uint8(out * 255)

        # cv.imshow("gasses", out)

        return out

    # Read image

    img = cv2.imread(filePath)

    # 添加椒盐噪声，噪声比例为 0.02

    out1 = sp_noise(img, prob=0.02)

    # 使用5*5的中值滤波器滤除椒盐噪声

    median = cv2.medianBlur(out1, 5)

    # 添加高斯噪声，均值为0，方差为0.001

    out2 = gasuss_noise(img, mean=0, var=0.004)

    # 去高斯噪声

    result = cv2.bilateralFilter(out2, 55, 100, 100)  # 双边滤波器

    result1 = cv2.fastNlMeansDenoisingColored(out2, None, 8, 8, 7, 21)  # 函数

    means5 = cv2.blur(out2, (5, 5))  # 均值滤波器

    # 显示图像

    plt.figure(1)

    plt.subplot(231)

    plt.axis('off')  # 关闭坐标轴

    plt.title('Original')

    plt.imshow(img[:, :, ::-1])

    plt.subplot(232)

    plt.axis('off')

    plt.title('Add Salt and Pepper noise')

    plt.imshow(out1[:, :, ::-1])

    plt.subplot(233)

    plt.axis('off')

    plt.title('Add Gaussian noise')

    plt.imshow(out2[:, :, ::-1])

    plt.subplot(235)

    plt.axis('off')

    plt.title('remove Salt and Pepper noise[zhongzhi]')

    plt.imshow(median[:, :, ::-1])

    plt.subplot(236)

    plt.axis('off')

    plt.title('remove Gaussian noise[hanshu]')

    plt.imshow(result1[:, :, ::-1])

    plt.subplot(234)

    plt.axis('off')

    plt.title('remove Gaussian noise[junzhi]')

    plt.imshow(means5[:, :, ::-1])

    plt.show()
