import cv2 as cv
import numpy as np

def addnoise(path):
    src = cv.imread(path)
    copy = np.copy(src)
    cv.imshow("input",src)
    h,w = src.shape[:2]  #获取图像的宽高信息
    nums = 5000
    rows = np.random.randint(0, h, (5000), dtype =int)
    cols = np.random.randint(0,w,(5000),dtype = int)
    for i in range(nums):
        if i%2 == 1:
            src[rows[i],cols[i]] = (255,255,255)
        else:src[rows[i],cols[i]] = (0,0,0)
    cv.imshow("salt and pepper image", src)
    gnoise = np.zeros(src.shape,src.dtype)
    m=(15,15,15)  #噪声均值
    s=(30,30,30) #噪声方差
    cv.randn(gnoise,m,s)##产生高斯噪声
    cv.imshow("gnoise",gnoise)#
    dst = cv.add(copy,gnoise)#将高斯噪声图像加到原图上去
    cv.imshow("gaussion",dst)



if __name__ == '__main__':
    addnoise()
    cv.waitKey(0)