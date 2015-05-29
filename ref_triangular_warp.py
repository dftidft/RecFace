import numpy as np
from landmark_data import *
from shape_model import *


def scale_landmark(landmark, size):
    n = landmark.size
    xs = [landmark[i] for i in range(n) if i % 2 == 0]
    ys = [landmark[i] for i in range(n) if i % 2 == 1]
    xs = [min(size[1] - 1, max(0, (x - min(xs)) * size[1] / (max(xs) - min(xs)))) for x in xs]
    ys = [min(size[0] - 1, max(0, (y - min(ys)) * size[0] / (max(ys) - min(ys)))) for y in ys]
    ret = []
    for i in range(n/2):
        ret.append((int(xs[i]), int(ys[i])))
    return ret


def warp_from_triangle(img, src_tri, des_tri):
    warp_mat = cv2.getAffineTransform(src_tri, des_tri)
    warped_img = cv2.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))
    return warped_img


size = (480, 640)

landmark_data = LandmarkData()
landmark = landmark_data.get_landmark('i000qe-fn')
img = landmark_data.get_img('i000qe-fn.jpg')
template_landmark = ShapeModel().subspace[:, 0]
template_landmark = scale_landmark(template_landmark, img.shape)

subdiv2d = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
for pts in landmark:
    subdiv2d.insert(pts)
src_tri_list = subdiv2d.getTriangleList()
subdiv2d = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
for pts in template_landmark:
    subdiv2d.insert(pts)
des_tri_list = subdiv2d.getTriangleList()

final_img = np.zeros(img.shape)
for i in range(len(src_tri_list)):
    src_tri = np.zeros((3, 2), dtype=np.float32)
    for j in range(3):
        src_tri[j, 0] = src_tri_list[i][j * 2]
        src_tri[j, 1] = src_tri_list[i][j * 2 + 1]
    des_tri = np.zeros((3, 2), dtype=np.float32)
    mask_tri = []
    for j in range(3):
        des_tri[j, 0] = des_tri_list[i][j * 2]
        des_tri[j, 1] = des_tri_list[i][j * 2 + 1]
        mask_tri.append((des_tri_list[i][j * 2], des_tri_list[i][j * 2 + 1]))

    print src_tri, des_tri
    warped_img = warp_from_triangle(img, src_tri, des_tri)
    #warp_mask = np.zeros((img.shape[0], img.shape[1]))
    #cv2.fillConvexPoly(warp_mask, des_tri, (255, 255, 255), cv2.LINE_AA, 0)
    #np.copyto(final_img, warped_img, where=warp_mask)
    print final_img.shape
    print warped_img.shape
    cv2.imshow('', warped_img)
    cv2.waitKey()
