# coding:utf-8
"""
利用 Procrustes Analysis 方法，把所有训练数据中的人脸特征点位置进行对齐
Procrustes Analysis 方法的主要过程如下：
循环以下过程 nIter 轮：
    1. 获取各特征点的平均位置
    2. 假定所有人脸都可以通过某种仿射变换对齐平均位置，利用最小二乘法计算变换矩阵，并得到每张人脸对齐后的新位置。
       其中最小二乘法可以直接用公式解析计算。
    3. 获取各特征点新的平均位置，如果新老平均位置非常相似，结束循环，否则回到第二步。
"""

from load_train_data import *
import matplotlib.pyplot as plt
nIter = 100
tolerance = 0.000001

train_data = load_train_data()

train_data = np.array(train_data)
num_data, dim_data = train_data.shape
# print train_data.shape

# 每张人脸的特征点减去人脸(特征点)中心位置
for i in range(num_data):
    mx = 0
    my = 0
    for j in range(dim_data / 2):
        mx += train_data[i, j * 2]
        my += train_data[i, j * 2 + 1]
    mx /= (dim_data / 2)
    my /= (dim_data / 2)
    for j in range(dim_data / 2):
        train_data[i, j * 2] -= mx
        train_data[i, j * 2 + 1] -= my

# Procrustes Analysis
# C: 每个特征点的平均位置
C_old = None
for iIter in range(nIter):
    C = train_data.mean(0)
    C = C / np.linalg.norm(C)
    # C = cv2.normalize(C, C)
    if iIter > 0 and np.linalg.norm(C - C_old) < tolerance:
        break
    print C
    C_old = C
    for i in range(num_data):
        pts = train_data[i, :]
        d = 0
        a = 0
        b = 0
        for j in range(pts.size / 2):
            d += pts[j * 2] * pts[j * 2] + pts[j * 2 + 1] * pts[j * 2 + 1]
            a += pts[j * 2] * C[j * 2] + pts[j * 2 + 1] * C[j * 2 + 1]
            b += pts[j * 2] * C[j * 2 + 1] - pts[j * 2 + 1] * C[j * 2]
        a /= d
        b /= d
        for j in range(pts.size / 2):
            x = train_data[i, j * 2]
            y = train_data[i, j * 2 + 1]
            train_data[i, j * 2] = a * x - b * y
            train_data[i, j * 2 + 1] = b * x + a * y

print iIter

np.save('shape.npy', train_data)

x = []
y = []
C = train_data.mean(0)
m = - C
for j in range(dim_data / 2):
    x.append(m[j * 2])
    y.append(m[j * 2 + 1])
plt.plot(x, y, 'r.')
plt.show()