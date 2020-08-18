import os.path
import matplotlib.pyplot as plt
import numpy as np

from mppca.mixture_ppca import MPPCA

n_samples = 500

theta = np.random.uniform(-np.pi, np.pi, size=n_samples)
x_1 = np.sin(theta)
x_2 = np.cos(theta)
r = np.random.normal(scale=0.1, size=n_samples) + 1.
X = np.array([x_1 * r, x_2 * r]).T

circle_mppca_path = "examples/circle_mppca.npz"
if os.path.exists(circle_mppca_path):
    mppca = MPPCA.load(circle_mppca_path)
else:
    mppca = MPPCA(10, 1, n_iterations=10)
    mppca.fit(X)
    mppca.save(circle_mppca_path)

Z = np.zeros((n_samples, 2))
c = np.zeros(n_samples)
for i in range(n_samples):
    c[i], latent = mppca.sample_latent()
    Z[i] = mppca.sample_from_latent(int(c[i]), latent, True)

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

    Z = np.array([mppca.reconstruction(np.array([x_1]), np.array([0])) for _ in range(10)])

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


