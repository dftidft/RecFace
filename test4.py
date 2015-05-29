import cv2
import numpy as np

mask_tri = np.array([[138, 0], [125, 43], [127, 7]], dtype=np.int32)

mask_img = np.zeros((640, 480, 3), dtype=np.uint8)
cv2.fillConvexPoly(mask_img, mask_tri, (255, 255, 255))
