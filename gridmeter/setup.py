#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This setup.py inspired by https://github.com/kennethreitz/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree
from setuptools import setup, find_packages, Command

OFF_DATAFLOW_REQUIRES = []

INSTALL_REQUIRES = [
    "attrs",
    "numpy>=1.24.4",
    "scipy>=1.10.1",
    "pandas>=1.1.0",
    "numba",
    "scikit-learn>=1.3.0",
    "fdasrsf>=2.4.1,<=2.5.2", # library broken on higher versions. Issue tracked here https://github.com/jdtuck/fdasrsf_python/issues/41
    "scikit-fda",
    "pydantic>=2.0.0",
    "eval_type_backport",
]
EXTRAS_REQUIRE = {"off-dataflow": OFF_DATAFLOW_REQUIRES}

here = os.path.abspath(os.path.dirname(__file__))
NAME = "gridmeter"
about = {}

with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        sys.exit()

setup(
    name=NAME,
    version=about["__version__"],
    description=about["__description__"],
    long_description=long_description,
    url=about["__url__"],
    author=about["__author__"],
    packages=find_packages(exclude=("tests",)),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    # $ setup.py publish support.
    cmdclass={"upload": UploadCommand},
)
