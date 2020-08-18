from setuptools import Extension, setup

from Cython.Build import cythonize
ext_modules = [
    Extension(
        "cmixture",
        ["cmixture.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]
setup(
    ext_modules=cythonize(ext_modules)
)