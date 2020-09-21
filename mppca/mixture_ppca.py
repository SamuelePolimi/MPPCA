"""
Mixture of Probabilistic Principal Component Analysers
M. tipping and C. bishop, 1999.

Adapted with numerical stabilization techniques
"""

import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn.covariance import ledoit_wolf
from mppca.multiprocess import MultiProcess
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mppca.dict_serializable import DictSerializable

from mppca.cmixture.cmixture import sum_logs, get_covariance, get_log_pi,\
    get_mean,  get_sigma_linear_tr, _update_responsabilities


def get_s(X, log_responsabilities, log_pi, means, component):
    m = X - means[component]
    tot_sum = (m * np.exp(log_responsabilities[component, :, np.newaxis])).T @ m
    return tot_sum / X.shape[0] / np.exp(log_pi[component])             # eq 24


def get_w_sigma(W_old, sigma_squared_old, S, iter=1):
    d = W_old.shape[1]
    q = W_old.shape[2]
    W = np.copy(W_old)
    W_new = np.zeros_like(W)
    sigma_squared = np.copy(sigma_squared_old)
    for it in range(iter):
        for j in range(W_old.shape[0]):
            M_inv = np.linalg.inv(sigma_squared[j] * np.eye(q) + W[j].T @ W[j])
            W_new[j] = S[j] @ W[j] @ np.linalg.inv(sigma_squared[j] * np.eye(q) + M_inv @ W[j].T @ S[j] @ W[j])
            sigma_squared[j] = np.trace(S[j] - S[j] @ W[j] @ M_inv @ W_new[j].T)/d
            W[j] = np.copy(W_new[j])
    return W, sigma_squared


class MPPCA(DictSerializable):

    load_fn = DictSerializable.get_numpy_load()

    def __init__(self, n_components: int, latent_dimension: int, n_iterations=100, tolerance=1E-5, cov_reg=1E-8, n_init=100):
        """

        :param components: Number of components of the mixture model
        :param latent_dimension: Number of latent dimension
        """

        self.n_components = n_components
        self.latent_dimension = latent_dimension
        self._n_iterations = n_iterations
        self._initialized = False
        self._tolerance = tolerance
        self._cov_reg = cov_reg
        self._n_init = n_init
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())

    @staticmethod
    def load_from_dict(**kwargs):
        mppca = MPPCA(kwargs["n_components"], kwargs["latent_dimension"], kwargs["n_iterations"],
                      n_init=kwargs["_n_init"])
        mppca.log_pi = kwargs["log_pi"]
        mppca.sigma_squared = kwargs["sigma_squared"]
        mppca.means = kwargs["means"]
        mppca.covariances = kwargs["covariances"]
        mppca.linear_transform = kwargs["linear_transform"]
        mppca._initialized = kwargs["_initialized"]
        return mppca

    @staticmethod
    def load(file_name: str):
        """

        :param file_name:
        :param domain:
        :return:
        """
        file = MPPCA.load_fn(file_name)
        return MPPCA.load_from_dict(**file)

    def _get_dict(self):
        return dict(n_components=self.n_components,
                    latent_dimension=self.latent_dimension,
                    n_iterations=self._n_iterations,
                    log_pi=self.log_pi,
                    sigma_squared=self.sigma_squared,
                    means=self.means,
                    covariances=self.covariances,
                    linear_transform=self.linear_transform,
                    _initialized=self._initialized,
                    _n_init=self._n_init
                    )

    def _reset(self, X):
        n_samples, observed_dimensions = X.shape
        kmeans = KMeans(self.n_components, n_init=self._n_init)
        lab = kmeans.fit(X).predict(X)
        self.covariances = []
        for i in range(self.n_components):
            cl_indxs = np.where(lab == i)[0]
            rnd_indxs = np.random.choice(range(n_samples), size=5)
            indx = np.concatenate([cl_indxs, rnd_indxs])
            # Avoid non-singular covariance
            self.covariances.append(ledoit_wolf(X[indx])[0])
        self.pi = np.ones(self.n_components)/self.n_components
        self.log_pi = np.log(self.pi)
        self.means = np.array(kmeans.cluster_centers_)

        self.linear_transform = np.random.uniform(size=(self.n_components, observed_dimensions, self.latent_dimension))
        for i in range(self.n_components):
            self.linear_transform[i] = np.eye(observed_dimensions, self.latent_dimension)
        self.sigma_squared = np.ones(self.n_components)
        self.covariances = np.array(self.covariances)

    def _fit(self, X):

        if not self._initialized:
            self._reset(X)
        for _ in range(self._n_iterations):
            pi, mu, W, sigma2, R, L, sigma2hist = mppca_gem(X, np.exp(self.log_pi), self.means, self.linear_transform, self.sigma_squared, 10)
            self.log_pi = np.log(pi)
            self.means = mu
            self.sigma_squared = sigma2
            self.linear_transform = W

    def fit(self, X):


        log_likelihood = -np.inf
        if len(X.shape) != 2:
            raise Exception("The shape of X must be of two dimensions.")

        mp = MultiProcess(n_process=23, backend="threading")

        if not self._initialized:
            self._reset(X)

        for t in range(self._n_iterations):
            state = np.copy(self.means), np.copy(self.covariances), np.copy(self.linear_transform), np.copy(
                self.sigma_squared), np.copy(self.log_pi)
            try:
                self.log_responsabilities, self.log_likelihoods = _update_responsabilities(X, self.n_components, self.means,
                                                                                           self.covariances, self.log_pi)
                self.log_pi = get_log_pi(self.log_responsabilities)

                args = [[i] for i in range(self.n_components)]
                kw_args = [{} for i in range(self.n_components)]

                self.means = np.array(mp.compute(lambda i: get_mean(X, self.log_responsabilities, i), args, kw_args,
                                                 check_pickle=False))

                start = time.time()
                S = np.array(mp.compute(lambda i: get_s(X, self.log_responsabilities, self.log_pi, self.means, i),
                                        args, kw_args, check_pickle=False))

                # print(start - time.time())

                res = mp.compute(lambda i: get_sigma_linear_tr(S, self.latent_dimension, i), args, kw_args,
                                 check_pickle=False)

                for j in range(self.n_components):
                    self.sigma_squared[j], self.linear_transform[j] = res[j]

                stab_inv = self._cov_reg
                self.sigma_squared = np.clip(self.sigma_squared, a_min=stab_inv, a_max=np.inf)
                self.covariances = np.array([get_covariance(w, s)
                                             for w, s in zip(self.linear_transform,
                                                            self.sigma_squared)])

                self._initialized = True

                current_log_likelihood = self.log_likelihood()
                if np.abs(current_log_likelihood - log_likelihood) <= self._tolerance:
                    return
                log_likelihood = current_log_likelihood
                print(log_likelihood)

            except:
                self.means, self.covariances, self.linear_transform, self.sigma_squared, self.log_pi = state
                break

    def _fit_backup(self, X):

        log_likelihood = -np.inf
        if len(X.shape) != 2:
            raise Exception("The shape of X must be of two dimensions.")

        mp = MultiProcess(n_process=23, backend="threading")

        if not self._initialized:
            self._reset(X)

        for t in range(self._n_iterations):

            state = np.copy(self.means), np.copy(self.covariances), np.copy(self.linear_transform), np.copy(self.sigma_squared), np.copy(self.log_pi)
            try:
                self.log_responsabilities, self.log_likelihoods = _update_responsabilities(X, self.n_components, self.means,
                                                                                           self.covariances, self.log_pi)

                self.log_pi = get_log_pi(self.log_responsabilities)

                args = [[i] for i in range(self.n_components)]
                kw_args = [{} for i in range(self.n_components)]
                self.means = np.array(mp.compute(lambda i: get_mean(X, self.log_responsabilities, i), args, kw_args,
                                        check_pickle=False))

                start = time.time()
                S = np.array(mp.compute(lambda i: get_s(X, self.log_responsabilities, self.log_pi, self.means, i),
                               args, kw_args, check_pickle=False))

                # print(start - time.time())

                res = mp.compute(lambda i: get_sigma_linear_tr(S, self.latent_dimension, i), args, kw_args,
                                 check_pickle=False)

                for j in range(self.n_components):
                    self.sigma_squared[j], self.linear_transform[j] = res[j]

                # self.linear_transform, self.sigma_squared = get_w_sigma(self.linear_transform, self.sigma_squared, S,
                #                                                         iter=1)

                stab_inv = 1E-10 #1E- # set to some little positive number e.g., 1E-10 to stabilize the matrix inversion
                self.sigma_squared = np.clip(self.sigma_squared, a_min=stab_inv, a_max=np.inf)
                self.covariances = np.array([get_covariance(w, s)
                                             for w, s in zip(self.linear_transform,
                                                             self.sigma_squared)])

                self._initialized = True

                current_log_likelihood = self.log_likelihood()
                if np.abs(current_log_likelihood - log_likelihood) <= self._tolerance:
                    return
                log_likelihood = current_log_likelihood

            except:
                self.means, self.covariances, self.linear_transform, self.sigma_squared, self.log_pi = state
                break
            for i in range(self.n_components):
                if linalg.cond(self.covariances[i]) > 1/sys.float_info.epsilon:
                    print("singularity detected")
                    self.means, self.covariances, self.linear_transform, self.sigma_squared, self.log_pi = state
                    return


    def log_likelihood(self):
        return np.mean(sum_logs(np.array([pi + p for pi, p in zip(self.log_pi, self.log_likelihoods)]), axis=0)) # eq 20

    def sample(self, noise=True):
        """
        Return a sample from the dataset
        Sample in the observed space.
        :param noise: Add isotropic noise.
        :return:
        """
        i = np.random.choice([j for j in range(self.n_components)], p=np.exp(self.log_pi))
        if noise:
            return np.random.multivariate_normal(self.means[i], self.covariances[i])
        else:
            return np.random.multivariate_normal(self.means[i], self.covariances[i] - \
                                                 self.sigma_squared[i]*np.eye(self.covariances[i].shape[0]))

    def sample_latent(self):
        """
        sample from the
        :return:
        """
        return int(np.random.choice(range(self.n_components), p=np.exp(self.log_pi))), \
               np.random.multivariate_normal(np.zeros(self.latent_dimension), np.eye(self.latent_dimension))

    def sample_from_latent(self, n_cluster, latent, noise=True):
        mean = self.linear_transform[n_cluster] @ latent + self.means[n_cluster]
        perturbation = 0. if not noise else np.random.multivariate_normal(np.zeros_like(mean),
                                                    self.sigma_squared[n_cluster]*np.eye(mean.shape[0]))
        return mean + perturbation

    def get_responsabilities(self, X: np.ndarray, idxs: np.ndarray, log_pi_lat, mean_lat, cov_lat):
        """

        :param X: Data
        :param n_components: number of the component of the mixture model
        :param means: means of the clusters
        :param covariances: covariances of the clusters
        :param log_pi: log weights of the mixture model
        :return: log_responsabilities, log_likelihood (both sample-wise)
        """

        obs_dim = self.means[0].shape[0]
        R_log = np.zeros(self.n_components)
        P_log = np.zeros(self.n_components)

        for i in range(self.n_components):
            mean = self.linear_transform[i] @ mean_lat[i] + self.means[i]
            cov = self.sigma_squared[i] * np.eye(obs_dim) + self.linear_transform[i] @ cov_lat[i] @ self.linear_transform[i].T
            P_log[i] = multivariate_normal.logpdf(X, mean[idxs], cov[idxs][:, idxs], allow_singular=True)

        log_scaling = sum_logs(np.array([P_log[j] + log_pi_lat[j]
                                         for j in range(self.n_components)]), axis=0)

        for i in range(self.n_components):
            R_log[i] = P_log[i] + log_pi_lat[i] - log_scaling  # eq 21


        return R_log, P_log

    def reconstruction(self, X, idx, use_mean_latent=False, noise=True, log_pi_lat=None, mu_lats=None, cov_lats=None):

        if log_pi_lat is None:
            log_pi_lat = self.log_pi

        old_log_pi = np.copy(self.log_pi)

        if mu_lats is None:
            mu_lats = np.zeros((self.n_components, self.latent_dimension))
        if cov_lats is None:
            cov_lats = [np.eye(self.latent_dimension) for i in range(self.n_components)]

        r, p = self.get_responsabilities(X, idx, log_pi_lat, mu_lats, cov_lats)

        cluster = np.random.choice(range(self.n_components), p=np.exp(r))

        mu_lat = mu_lats[cluster]
        cov_lat = cov_lats[cluster]

        W = self.linear_transform[cluster]
        mean = self.means[cluster]
        obs_dimension = mean.shape[0]
        sigma_sq = self.sigma_squared[cluster]

        if noise:
            cov_bb = W @ cov_lat @ W.T + sigma_sq * np.eye(obs_dimension)
        else:
            cov_bb = W @ cov_lat @ W.T

        mu_c = np.concatenate([mu_lat, mean])
        cov_c = np.concatenate(
            [np.concatenate([cov_lat, cov_lat @ W.T], axis=1),
            np.concatenate([W @ cov_lat, cov_bb], axis=1)], axis=0
        )

        idx_c = self.latent_dimension + idx
        idx_query = np.array([i for i in range(mean.shape[0] + self.latent_dimension) if i not in idx_c])
        # we regulize with sigma... i.e., the context we assumed is observed with the presence of noise, but
        # the reconstruction will be noiseless
        mu_a, cov_a = self.conditional(mu_c, cov_c, X, idx_c, idx_query, reg_cov_bb=sigma_sq)
        if use_mean_latent:
            x_sample = mu_a
        else:
            x_sample = np.random.multivariate_normal(mu_a, cov_a)

        ret = np.zeros(mean.shape[0] + self.latent_dimension)
        ret[idx_query] = x_sample
        ret[idx_c] = X

        # W_sq = W.T @ W
        # M_inv = np.linalg.inv(self.sigma_squared[cluster]*np.eye(W_sq.shape[0]) + W_sq)
        # new_val = self.means[cluster].copy()
        # new_val[idx] = X
        # mean_latent = M_inv @ W.T @ (new_val - self.means[cluster])
        # cov_latent = self.sigma_squared[cluster] * M_inv
        # if use_mean_latent:
        #     latent = mean_latent
        # else:
        #     latent = np.random.multivariate_normal(mean_latent, cov_latent)
        # res = self.sample_from_latent(cluster, latent, noise=noise)
        # # overriding the sample is inconsistent with the model
        # # res[idx] = X
        return ret[self.latent_dimension:], cluster, ret[:self.latent_dimension]

    def conditional(self, mean, covariance, X, idx, idx_query, reg_cov_bb=0.):
        indx_a = idx_query
        indx_b = idx
        mu_a = mean[indx_a]
        mu_b = mean[indx_b]
        cov_aa = covariance[indx_a][:, indx_a]
        cov_ab = covariance[indx_a][:, indx_b]
        cov_bb = covariance[indx_b][:, indx_b] + reg_cov_bb * np.eye(len(idx))

        inv_cov_bb = np.linalg.inv(cov_bb)
        mu_a_b = mu_a + cov_ab @ inv_cov_bb @ (X - mu_b)

        return mu_a_b, cov_aa - cov_ab @ inv_cov_bb @ cov_ab.T