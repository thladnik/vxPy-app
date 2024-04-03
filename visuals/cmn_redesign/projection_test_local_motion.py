from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

plt.figure()
ax = plt.subplot(projection='3d')

centers = np.random.rand(300, 3) - 0.5
centers /= np.linalg.norm(centers, axis=1)[:,None]

motvec = np.random.rand(3) - 0.5

ax.scatter(*centers.T, s=10)
ax.quiver(*[0, 0, 0], *motvec, color='black')

motvecs_local = []
for center in centers:
    motveclocal = motvec - center * np.dot(motvec, center)
    motvecs_local.append(motveclocal)

motvecs_local = np.array(motvecs_local)

ax.quiver(*centers.T, *motvecs_local.T / 2, color='gray')

ax.set_aspect('equal')

plt.show()

