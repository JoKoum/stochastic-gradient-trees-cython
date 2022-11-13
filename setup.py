from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "SGTWrapper",
        ["SGTWrapper.pyx", "cysgt/utils/StochasticGradientTree.cpp"],
        language="c++"
    )
]

setup(
    name="cysgt",
    version="0.0.1",
    description="Stochastic Gradient Trees implementation using Cython",
    long_description="Stochastic Gradient Trees by Henry Gouk, Bernhard Pfahringer, and Eibe Frank implementation in C++, Python. Based on the parer's accompanied repository code.",
    url="https://github.com/JoKoum/stochastic-gradient-trees-cython",
    author="John Koumentis",
    author_email="jokoum92@gmail.com",
    license="MIT",
    packages=['cysgt','cysgt.utils'],
    package_dir = {"cysgt": "cysgt"},
    install_requires=["numpy>=1.20.2", "cython>=0.29.28", "pandas>=1.3.3", "scikit-learn>=0.24.2"],
    zip_safe=False
)