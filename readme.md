Mixture of Probabilistic Principal Component Analysis (MPPCA)
=============================================================

Installation
------------

First, copy the content of the repository in your favourite location

```console
git clone fuffa
cd MPPCA
```

then navigate in ``cmixture`` and compile the ``cython`` script

```console
cd mppca/cmixture
python setup.py build_ext --inplace
```

Lastly, install the library

```console
cd ../..
pip install -e .
```

A fisrt example with a circular data:






