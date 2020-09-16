import os
from setuptools import setup, find_packages

OFF_DATAFLOW_REQUIRES = []

INSTALL_REQUIRES = ["pandas>=1.1.0", "plotnine"]
EXTRAS_REQUIRE = {"off-dataflow": OFF_DATAFLOW_REQUIRES}

here = os.path.abspath(os.path.dirname(__file__))
NAME = "eesampling"
about = {}

with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

setup(
    name="eesampling",
    version=about["__version__"],
    description=about["__description__"],
    url=about["__url__"],
    author=about["__author__"],
    classifiers=[],
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
