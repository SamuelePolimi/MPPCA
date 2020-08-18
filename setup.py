from setuptools import setup

setup(name='mppca',
      version='0.1.0',
      description='Mixture of Probabilistic Principal Component Analysis.',
      author='Intelligent Autonomous Systems Lab',
      author_email='samuele.tosatto@tu-darmstadt.de',
      license='MIT',
      packages=['mppca'],
      zip_safe=False,
      install_requires=[
          'numpy>=1.18.1',
          'scipy>=1.4.1',
          'scikit-learn>=0.23.1',
          'matplotlib>=3.1.2'
      ])