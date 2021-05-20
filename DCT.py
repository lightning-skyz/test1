import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

# 读取图片

def DCT1(filePath):
    image = cv2.imread(filePath, 0)
    # 离散余弦变换，并获取其幅频谱
    img_dct = cv2.dct(np.float32(image))
    img_dct_log = 20 * np.log(abs(img_dct))
    # 逆离散余弦变换，变换图像至空间域
    img_back = cv2.idct(img_dct)
    print(type(img_dct))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('original ', fontproperties='Times New Roman')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_dct_log, cmap='gray')
    plt.title('DCT ', fontproperties='Times New Roman')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_back, cmap='gray')
    plt.title('iDCT ', fontproperties='Times New Roman')
    plt.axis('off')

    plt.show()

    plt.close()

    image_dct_1 = copy.deepcopy(img_dct)
    image_dct_2 = copy.deepcopy(img_dct)
    image_dct_3 = copy.deepcopy(img_dct)

    # 此处为显示
    image_dct_log_1 = 20 * np.log(abs(image_dct_1))
    image_dct_log_2 = 20 * np.log(abs(image_dct_2))
    image_dct_log_3 = 20 * np.log(abs(image_dct_3))

    # 裁剪DCT
    for i in range(image_dct_1.shape[0]):
        for j in range(image_dct_1.shape[1]):
            if i > (256 + 128) or j > (256 + 128):
                image_dct_1[i, j] = 0
                image_dct_log_1[i, j] = 0
            if i > 256 or j > 256:
                image_dct_2[i, j] = 0
                image_dct_log_2[i, j] = 0
            if i > 128 or j > 128:
                image_dct_3[i, j] = 0
                image_dct_log_3[i, j] = 0

    # 逆DCT
    img_back_1 = cv2.idct(image_dct_1)
    img_back_2 = cv2.idct(image_dct_2)
    img_back_3 = cv2.idct(image_dct_3)

    plt.subplot(2, 3, 1)
    plt.imshow(image_dct_log_1, cmap='gray')
    plt.title('compression 1', fontproperties='Times New Roman')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(img_back_1, cmap='gray')
    plt.title(' IDCT 1', fontproperties='Times New Roman')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(image_dct_log_2, cmap='gray')
    plt.title('compression 2', fontproperties='Times New Roman')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(img_back_2, cmap='gray')
    plt.title(' IDCT 2', fontproperties='Times New Roman')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(image_dct_log_3, cmap='gray')
    plt.title('compression 3', fontproperties='Times New Roman')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(img_back_3, cmap='gray')
    plt.title(' IDCT 3', fontproperties='Times New Roman')
    plt.axis('off')

    plt.show()