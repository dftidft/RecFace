# coding:utf-8
# 把原始的平均shape（有正有负）映射到一个固定的图片区域

import cv2
import numpy as np
from landmark_data import *
from shape_model import *

size = (480, 640)
img = np.zeros(size)
landmark = ShapeModel().subspace[:, 0]
n = landmark.size
xs = [landmark[i] for i in range(n) if i % 2 == 0]
ys = [landmark[i] for i in range(n) if i % 2 == 1]

xs = [(x - min(xs)) * size[1] / (max(xs) - min(xs)) for x in xs]
ys = [(y - min(ys)) * size[0] / (max(ys) - min(ys)) for y in ys]

for i in range(n/2):
    cv2.circle(img, (int(xs[i]), int(ys[i])), 1, (255, 255, 255))

cv2.imshow('', img)
cv2.waitKey()

#print landmark
#print xs