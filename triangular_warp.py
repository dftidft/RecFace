import numpy as np
from landmark_data import *
from shape_model import *


def scale_landmark(template_landmark, size):
    n = template_landmark.size
    xs = [template_landmark[i] for i in range(n) if i % 2 == 0]
    ys = [template_landmark[i] for i in range(n) if i % 2 == 1]
    xs = [min(size[1] - 1, max(0, (x - min(xs)) * size[1] / (max(xs) - min(xs)))) for x in xs]
    ys = [min(size[0] - 1, max(0, (y - min(ys)) * size[0] / (max(ys) - min(ys)))) for y in ys]
    ret = []
    for i in range(n/2):
        ret.append((int(xs[i]), int(ys[i])))
    return ret


def warp_from_triangle(img, src_tri, des_tri, size):
    warp_mat = cv2.getAffineTransform(src_tri, des_tri)
    warped_img = cv2.warpAffine(img, warp_mat, (size[1], size[0]))
    return warped_img


def get_pt_index(pt, pts):
    for i in range(len(pts)):
        if pts[i][0] == pt[0] and pts[i][1] == pt[1]:
            return i
    return -1


def draw_triangles(img, tri_indices, landmark):
    for tri_index in tri_indices:
        for i in range(3):
            cv2.line(img, landmark[tri_index[i % 3]], landmark[tri_index[(i + 1) % 3]], (0, 255, 0), 1)
    cv2.imshow('', img)
    cv2.waitKey()


size = (640, 480, 3)

landmark_data = LandmarkData()
landmark = landmark_data.get_landmark('i012se-mn')
img = landmark_data.get_img('i012se-mn.jpg')
template_landmark = ShapeModel().subspace[:, 0]
template_landmark = scale_landmark(template_landmark, img.shape)

subdiv2d = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
for pts in landmark:
    subdiv2d.insert(pts)
src_tri_list = subdiv2d.getTriangleList()


tri_indices = []
for tri in src_tri_list:
    tri_index = []
    is_valid_point = True
    for i in range(3):
        x = tri[2 * i]
        y = tri[2 * i + 1]
        if x >= 0 and y >= 0 and x < img.shape[1] and y < img.shape[0]:
            tri_index.append(get_pt_index((x, y), landmark))
        else:
            is_valid_point = False
            break
    if is_valid_point:
        tri_indices.append(tri_index)



final_img = np.zeros(size, dtype=np.uint8)
for tri_idx in tri_indices:
    src_tri = np.zeros((3, 2), dtype=np.float32)
    des_tri = np.zeros((3, 2), dtype=np.float32)
    mask_tri = np.zeros((3, 2), dtype=np.int32)
    row = 0
    #print tri_idx
    for pt_idx in tri_idx:
        #print landmark[pt_idx]
        src_tri[row, 0] = landmark[pt_idx][0]
        src_tri[row, 1] = landmark[pt_idx][1]
        des_tri[row, 0] = template_landmark[pt_idx][0]
        des_tri[row, 1] = template_landmark[pt_idx][1]
        mask_tri[row, 0] = template_landmark[pt_idx][0]
        mask_tri[row, 1] = template_landmark[pt_idx][1]
        row += 1
    #print src_tri, des_tri

    warped_img = warp_from_triangle(img, src_tri, des_tri, size)
    mask = np.zeros(img.shape, np.bool)
    print mask_tri
    mask_img = np.zeros(size, dtype=np.uint8)
    cv2.fillConvexPoly(mask_img, mask_tri, (255, 255, 255))
    mask = np.array(mask_img, dtype=bool)
    np.copyto(final_img, warped_img, where=mask)


draw_triangles(final_img, tri_indices, template_landmark)
cv2.waitKey()
























