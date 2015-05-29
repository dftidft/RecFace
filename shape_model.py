import numpy as np

class ShapeModel:
    def __init__(self):
        self.model = np.load('shape_model.npz')
        self.subspace = self.model['subspace']
        self.subspace_std = self.model['subspace_std']

    def clamp(self, p, c=3.0):
        scale = p[0]
        for i in range(4, self.subspace_std.size):
            if (abs(p[i] / p[0]) > self.subspace_std[i]):
                p[i] = c * self.subspace_std[i] if p[i] > 0 else - c * self.subspace_std[i]

    def proj(self, shape):
        p = np.dot(self.subspace.transpose(), shape)
        self.clamp(p)
        return p

    def back_proj(self, p):
        return np.dot(self.subspace, p)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    shape_model = ShapeModel()
    shape_pts = np.load('shape.npy')

    x = []
    y = []
    px = []
    py = []
    m = shape_pts[0, :]
    pm = shape_model.proj(m)
    m = -m
    pm = - shape_model.back_proj(pm)
    for j in range(m.size / 2):
        x.append(m[j * 2])
        y.append(m[j * 2 + 1])
        px.append(pm[j * 2])
        py.append(pm[j * 2 + 1])
    plt.figure()
    plt.plot(x, y, 'r.')
    plt.plot(px, py, 'b.')
    plt.show()
