from sklearn.datasets import fetch_openml
from mppca.mixture_ppca import MPPCA
import matplotlib.pyplot as plt
import numpy as np

n_samples = 500

theta = np.random.uniform(-np.pi, np.pi, size=n_samples)
x_1 = np.sin(theta)
x_2 = np.cos(theta)
r = np.random.normal(scale=0.1, size=n_samples) + 1.
X = np.array([x_1 * r, x_2 * r]).T

# mmpca = MPPCA(10, 1, n_iterations=10)
# mmpca.fit(X)

# mmpca.save("examples/circle_mppca.npz")
mmpca = MPPCA.load("examples/circle_mppca.npz")

Z = np.zeros((n_samples, 2))
c = np.zeros(n_samples)
for i in range(n_samples):
    c[i], latent = mmpca.sample_latent()
    Z[i] = mmpca.sample_from_latent(int(c[i]), latent, True)

fig, axs = plt.subplots(1, 2)
axs[0].scatter(X[:, 0], X[:, 1])
axs[0].set_title("Dataset")
axs[1].scatter(Z[:, 0], Z[:, 1], c=c)
axs[1].set_title("Mixture of PPCA")
plt.show()

print("reconstruction")

for _ in range(10):
    theta_new = np.random.uniform(-np.pi, np.pi)
    x_1 = np.sin(theta_new)
    x_2 = np.cos(theta_new)
    r = np.random.normal(scale=0.1) + 1.
    X = np.array([x_1 * r, x_2 * r])

    Z = np.array([mmpca.reconstruction(np.array([x_1]), np.array([0])) for _ in range(10)])

    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(x_1, x_2)
    axs[0].set_title("Partial Observed Point")
    axs[0].set_xlim(-1.2, 1.2)
    axs[0].set_ylim(-1.2, 1.2)
    axs[1].scatter(Z[:, 0], Z[:, 1])
    axs[1].set_title("Reconstruction")
    axs[1].set_xlim(-1.2, 1.2)
    axs[1].set_ylim(-1.2, 1.2)
    plt.show()


