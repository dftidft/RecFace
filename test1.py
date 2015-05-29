# coding:utf-8
# 检验shapeModel的第0列与所有数据的平均shape之间的关系
# shapeModel的第0列 == normalize(平均shape)

from shape_model import *


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    shape_model = ShapeModel()
    shape_pts = np.load('shape.npy')

    x = []
    y = []
    m = - shape_model.subspace[:, 0]
    print m[0:10]
    m = - shape_pts.mean(0)
    print m[0:10] / np.linalg.norm(m)
    for j in range(m.size / 2):
        x.append(m[j * 2])
        y.append(m[j * 2 + 1])
    plt.figure()
    plt.plot(x, y, 'r.')
    plt.show()