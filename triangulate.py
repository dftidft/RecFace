import cv2
import numpy as np
from landmark_data import *

shape_pts = np.load('shape.npy')

landmark_data = LandmarkData()
landmark = landmark_data.get_landmark('i000qe-fn')
img = landmark_data.get_img('i000qe-fn.jpg')

subdiv2d = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))

for pts in landmark:
    subdiv2d.insert(pts)
for edge in subdiv2d.getEdgeList():
    pt = []
    out_bbox = False
    pt.append((edge[0], edge[1]))
    pt.append((edge[2], edge[3]))
    for i in range(2):
        if pt[i][0] < 0 or pt[i][1] < 0 or pt[i][0] > img.shape[1] or pt[i][1] > img.shape[0]:
            out_bbox = True
    if not out_bbox:
        cv2.line(img, pt[0], pt[1], (0, 255, 0), 1)

cv2.imshow('', img)
cv2.waitKey()
