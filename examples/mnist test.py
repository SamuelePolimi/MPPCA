from mppca.mixture_ppca import MPPCA
import matplotlib.pyplot as plt
import numpy as np

X = np.load("mnist.npy")

mppca = MPPCA(100, 10, n_iterations=10)

mppca.fit(X)
mppca.save("mnist_mppca.npz")

for _ in range(10):
    plt.matshow(mppca.sample(noise=False).reshape(28, 28))
    plt.show()

for _ in range(10):
    plt.matshow(mppca.sample(noise=True).reshape(28, 28))
    plt.show()