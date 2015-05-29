# coding:utf-8

"""
根据训练数据，计算人脸位置模型
计算思路大致如下：
    1. 位置信息太多，位置信息存在某些内在的联系，可以（或者说需要）进行特征分解，将原始数据映射到低维空间
    2. 人脸低维空间的特征由两部分信息组成：
        第一部分：假定人脸共有n维位置信息，如果我们承认任何一张人脸都应该是对齐后的平均人脸仿射变换得到，
            即只应该考虑所有能够由平均人脸仿射变换可以组成的所有点，而不是任意n维空间上的点，我们可以得到
            4个n维向量的线性组合描述这个子空间，4个线性组合的系数对应仿射变换的4个参数。另外对这4个向量做正交规范化。
        第二部分：将所有训练人脸位置都映射到第一部分描述的低维空间，残差构造一个协方差矩阵，做PCA分解，
            得到特征值比较大的几个特征向量。
        对于任意一个输入人脸位置，我们都把它映射到上述子空间，即先获得它仿射变换到平均脸的参数，再得到它的主要残差参数
    3. 限定一些不合理的人脸位置。即如果某个参数太离谱了，最多控制在该参数标准差的c倍以内，称为Clamp

注意：
    最终的V(model.subspace)的第0列中，就是原始的平均形状位置(x1, y1, ..., xn, yn)规范化后（等比例缩放）的结果，
    因为根据Gram-Schmidt算法，第0列是保持原状的，其他的列逐一保持与前i列正交
"""

import numpy as np
import matplotlib.pyplot as plt

# 协方差矩阵取特征向量的最大个数
max_num_eigens = 10
# 所取特征向量的特征值之和不少于矩阵总特征值的百分比
frac = 0.95

shape_pts = np.load('shape.npy')
C = shape_pts.mean(0)

N = shape_pts.shape[0]
n = C.size / 2
R = np.empty((n * 2, 4), np.float)
for i in range(n):
    R[i * 2, 0] = C[i * 2]
    R[i * 2 + 1, 0] = C[i * 2 + 1]
    R[i * 2, 1] = - C[i * 2 + 1]
    R[i * 2 + 1, 1] = C[i * 2]
    R[i * 2, 2] = 1.0
    R[i * 2, 3] = 0.0
    R[i * 2 + 1, 2] = 0.0
    R[i * 2 + 1, 3] = 1.0

# Gram-Schmidt Orthonormalization
for i in range(4):
    r = R[:, i]
    for j in range(i):
        b = R[:, j]
        r -= (r * b) * b
    R[:, i] = r / np.linalg.norm(r)

ds = shape_pts - np.dot(np.dot(shape_pts, R), R.transpose())

U, s, V = np.linalg.svd(np.dot(ds.transpose(), ds))

print U.shape, s.shape, V.shape

m = min(max_num_eigens, n - 1, N - 1)
sum_s = 0
for i in range(m):
    sum_s += s[i]
cur_s = 0
k_eigens = 0
for i in range(m):
    cur_s += s[i]
    if cur_s / sum_s > frac:
        break
    k_eigens += 1

print k_eigens

D = U[:, :k_eigens]

print D.shape

V = np.concatenate((R, D), axis=1)

print V.shape

Q = np.dot(V.transpose(), shape_pts.transpose())

print Q.shape

for i in range(N):
    v = Q[0, i]
    Q[:, i] /= v

e = Q.std(1)
e[0:4] = -1
print e

np.savez('shape_model.npz', subspace=V, subspace_std=e)

'''
proj = - np.dot(np.dot(shape_pts[0, :], V), V.transpose())
x = []
y = []
proj_x = []
proj_y = []
m = - C
for j in range(m.size / 2):
    x.append(m[j * 2])
    y.append(m[j * 2 + 1])
    proj_x.append(proj[j * 2])
    proj_y.append(proj[j * 2 + 1])
plt.figure()
plt.plot(x, y, 'r.')
plt.plot(proj_x, proj_y, 'b.')
plt.show()
'''