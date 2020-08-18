import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn.covariance import ledoit_wolf
from herl.multiprocess import MultiProcess
from scipy.stats import multivariate_normal

from mppca.dict_serializable import DictSerializable

from mppca.cmixture.cmixture import sum_logs, get_covariance, get_log_pi,\
    get_mean,  get_sigma_linear_tr, _update_responsabilities


def get_s(X, log_responsabilities, log_pi, means, component):
    m = X - means[component]
    tot_sum = (m * np.exp(log_responsabilities[component, :, np.newaxis])).T @ m
    return tot_sum / X.shape[0] / np.exp(log_pi[component])


class MPPCA(DictSerializable):

    load_fn = DictSerializable.get_numpy_load()

    def __init__(self, n_components: int, latent_dimension: int, n_iterations=100):
        """

        :param components: Number of components of the mixture model
        :param latent_dimension: Number of latent dimension
        """

        self.n_components = n_components
        self.latent_dimension = latent_dimension
        self._n_iterations = n_iterations
        self._initialized = False
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())

    @staticmethod
    def load_from_dict(**kwargs):
        mppca = MPPCA(kwargs["n_components"], kwargs["latent_dimension"], kwargs["n_iterations"])
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
                    _initialized=self._initialized
                    )

    def _reset(self, X):
        n_samples, observed_dimensions = X.shape
        kmeans = KMeans(self.n_components, random_state=0)
        lab = kmeans.fit(X).predict(X)
        self.covariances = []
        for i in range(self.n_components):
            indx = np.where(lab == i)[0]
            # Avoid non-singular covariance
            self.covariances.append(ledoit_wolf(X[indx])[0])
        self.pi = np.ones(self.n_components)/self.n_components
        self.log_pi = np.log(self.pi)
        self.means = np.array(kmeans.cluster_centers_)

        self.linear_transform = np.random.uniform(size=(self.n_components, observed_dimensions, self.latent_dimension))
        self.sigma_squared = np.ones(self.n_components)
        self.covariances = np.array(self.covariances)

    def fit(self, X):
        if len(X.shape) != 2:
            raise Exception("The shape of X must be of two dimensions.")

        mp = MultiProcess(n_process=23, backend="threading")

        if not self._initialized:
            self._reset(X)

        for t in range(self._n_iterations):

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

            self.covariances = np.array([get_covariance(w, s)
                                         for w, s in zip(self.linear_transform, self.sigma_squared)])

            self._initialized = True

            print(self.log_likelihood())

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

    def get_responsabilities(self, X: np.ndarray, idxs: np.ndarray):
        """

        :param X: Data
        :param n_components: number of the component of the mixture model
        :param means: means of the clusters
        :param covariances: covariances of the clusters
        :param log_pi: log weights of the mixture model
        :return: log_responsabilities, log_likelihood (both sample-wise)
        """

        R_log = np.zeros(self.n_components)
        P_log = np.zeros(self.n_components)

        for i in range(self.n_components):
            P_log[i] = multivariate_normal.logpdf(X, self.means[i, idxs], self.covariances[i, idxs][:, idxs])

        log_scaling = sum_logs(np.array([P_log[j] + self.log_pi[j]
                                         for j in range(self.n_components)]), axis=0)

        for i in range(self.n_components):
            R_log[i] = P_log[i] + self.log_pi[i] - log_scaling  # eq 21

        return R_log, P_log

    def reconstruction(self, X, idx, noise=True):
        r, p = self.get_responsabilities(X, idx)
        cluster = np.random.choice(range(self.n_components), p=np.exp(r))
        W = self.linear_transform[cluster]
        W_sq = W.T @ W
        M_inv = np.linalg.inv(self.sigma_squared[cluster]*np.eye(W_sq.shape[0]) + W_sq)
        new_val = self.means[cluster].copy()
        new_val[idx] = X
        mean_latent = M_inv @ W.T @ (new_val - self.means[cluster])
        cov_latent = self.sigma_squared[cluster] * M_inv
        res = self.sample_from_latent(cluster, np.random.multivariate_normal(mean_latent,
                                                                       cov_latent), noise=noise)
        # overriding the sample is inconsistent with the model
        # res[idx] = X
        return res