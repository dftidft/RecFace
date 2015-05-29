import numpy as np
import matplotlib.pyplot as plt

shape = np.load('shape.npy')

x = []
y = []
m = - shape.mean(0)
for j in range(m.size / 2):
    x.append(m[j * 2])
    y.append(m[j * 2 + 1])
plt.plot(x, y, 'r.')
plt.show()