"""
Mixture of Probabilistic Principal Component Analysers
M. tipping and C. bishop, 1999.

Adapted with numerical stabilization techniques
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from cython.parallel import prange

def sort_eigendecomp(mat):
    eigenValues, eigenVectors = np.linalg.eig(mat)
    eigenValues = np.real(eigenValues)
    eigenVectors = np.real(eigenVectors)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return eigenVectors, eigenValues


def truncated_eigendecomp(mat, k):
    e_vec, e_v = sort_eigendecomp(mat)
    return e_vec[:, :k], np.diag(e_v[:k]), np.mean(e_v[k:])


def sum_logs(X, axis=0, mul=1.):
    """
    X is in log-space, we will return the sum in log space
    :param X:
    :param axis:
    :return:
    """
    x_max = np.max(X, axis=axis)
    X_exp = mul * np.exp(X-x_max)
    return x_max + np.log(np.sum(X_exp, axis=axis))


def _update_responsabilities(X: np.ndarray, n_components: int, means: np.ndarray, covariances: np.ndarray,
                             log_pi: np.ndarray):
    """

    :param X: Data
    :param n_components: number of the component of the mixture model
    :param means: means of the clusters
    :param covariances: covariances of the clusters
    :param log_pi: log weights of the mixture model
    :return: log_responsabilities, log_likelihood (both sample-wise)
    """

    R_log = np.zeros((n_components, X.shape[0]))
    P_log = np.zeros((n_components, X.shape[0]))

    for i in range(n_components):
        P_log[i] = multivariate_normal.logpdf(X, means[i], covariances[i])

    log_scaling = sum_logs(np.array([P_log[j] + log_pi[j]
                                           for j in range(n_components)]), axis=0)

    for i in range(n_components):
        R_log[i] = P_log[i] + log_pi[i] - log_scaling       # eq 21

    return R_log, P_log


def get_log_pi(log_responsabilities: np.ndarray):
    """
    Get the log_weights of the mixture.
    :param log_responsabilities:
    :return:
    """
    n_samples = log_responsabilities.shape[1]
    return sum_logs(log_responsabilities.T, axis=0, mul=1. / n_samples).T   # eq 22


def get_mean(X, log_responsabilities, component):
    n_samples = log_responsabilities.shape[1]
    return np.sum([X[i] * np.exp(log_responsabilities[component, i])
            for i in range(n_samples)], axis=0) / np.exp(sum_logs(log_responsabilities[component]))


def get_s(X, log_responsabilities, log_pi, means, int component):

    tot_sum = 0.
    cdef Py_ssize_t i = 0
    cdef int n_samples = log_responsabilities.shape[1]
    for i in range(n_samples):
       tot_sum += exp(log_responsabilities[component, i]) * np.outer(X[i,:] -means[component], X[i,:] - means[component])
    return tot_sum / n_samples / np.exp(log_pi[component])                  # eq 14


def get_sigma_linear_tr(S, latent_dimension, component):
    U, L, sigma = truncated_eigendecomp(S[component], latent_dimension)

    return sigma, U @ sqrtm(L - sigma * np.eye(latent_dimension))           # eq 14


def get_covariance(linear_transform, sigma):
    return sigma * np.eye(linear_transform.shape[0]) + linear_transform @ linear_transform.T