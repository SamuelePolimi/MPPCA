"""
Copied from HeRL
"""

import numpy as np

from joblib import Parallel, delayed
from typing import Iterable, Callable

from herl.utils import Printable

import numpy as np

from joblib import Parallel, delayed
from typing import Iterable, Callable

from herl.utils import Printable


class MultiProcess(Printable):

    def __init__(self, n_process=20, random_seed=True, backend="multiprocessing"):
        self._n_process = n_process
        self._use_random_seed = random_seed
        self._backend = backend

    def compute(self, process: Callable, params=10, dict_args=None, check_pickle=True) -> Iterable:

        if type(params) is int:
            args_params = [[] for _ in range(params)]
        else:
            args_params = params

        if dict_args is None:
            kwargs_params = [{} for _ in range(params)]
        else:
            kwargs_params = dict_args

        def rnd_process(*args, **kwargs):
            if self._use_random_seed:
                np.random.seed()
            return process(*args, **kwargs)

        if self._backend == "none":
            return [rnd_process(*a, **k) for a, k in zip(args_params, kwargs_params)]

        return Parallel(n_jobs=self._n_process, backend=self._backend)(
            delayed(rnd_process)(*a, **k)
            for a, k in zip(args_params, kwargs_params))