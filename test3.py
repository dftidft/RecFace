# coding:utf-8
# 验证numpy的copyto函数用法

import numpy as np
import cv2

img = cv2.imread('d:/dataset/lena.jpg')
mask_img = np.zeros(img.shape, dtype=np.uint8)
mask = np.zeros(img.shape, np.bool)
#mask[0:255, 0:255, :] = True
pts = np.array([[0, 0], [0, 255], [255, 0]])
cv2.fillConvexPoly(mask_img, pts, (255, 255, 255))
mask = np.array(mask_img, dtype=bool)
np.copyto(mask_img, img, where=mask)
cv2.imshow('', mask_img)
cv2.waitKey()


