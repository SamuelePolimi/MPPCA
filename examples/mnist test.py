import os.path
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

from mppca.mixture_ppca import MPPCA

mnist_file_path = "examples/mnist.npy"
mnist_mppca_path = "examples/mnist_mppca.npz"

if os.path.exists(mnist_file_path):
    X = np.load(mnist_file_path)
else:
    X, _ = fetch_openml('mnist_784', version=1, return_X_y=True)
    X.save(mnist_file_path)

print("sample from the MNIST dataset.")
fig, axs = plt.subplots(1, 3)
indxs = np.random.randint(X.shape[0], size=3)
for i, indx in enumerate(indxs):
    axs[i].matshow(X[indx].reshape(28, 28))
    axs[i].set_title("Digit #%d" % indx)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.show()

if os.path.exists(mnist_mppca_path):
    mppca = MPPCA.load(mnist_mppca_path)
else:
    mppca = MPPCA(100, 10, n_iterations=10)
    mppca.fit(X)
    mppca.save(mnist_mppca_path)

print("Sample some digits from our MPPCA model.")

fig, axs = plt.subplots(1, 3)
for i in range(3):
    axs[i].matshow(mppca.sample(noise=True).reshape(28, 28))
    axs[i].set_title("Generated")
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.show()

print("Digit Reconstruction.")

fig, axs = plt.subplots(2, 3)
indxs = np.random.randint(X.shape[0], size=3)
for i, indx in enumerate(indxs):
    par_X = X[indx].copy()
    par_X[14*28:] = np.zeros((14, 28)).ravel() * np.nan
    axs[0, i].matshow(par_X.reshape((28, 28)))
    axs[0, i].set_title("Digit #%d" % indx)
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])
    rec_X = mppca.reconstruction(X[indx, :14*28], np.array(range(14*28)))
    axs[1, i].matshow(rec_X.reshape((28, 28)))
    axs[1, i].set_title("Sampled reconstruction")
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])

plt.show()
