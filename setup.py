#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   Copyright 2014-2023 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Standard library imports are grouped together at the top.
import io
import os
import sys
from shutil import rmtree

# Third-party imports: setuptools components are imported together.
from setuptools import find_packages, setup, Command

# Package metadata is centralized in one location for easy maintenance.
NAME = "eemeter"
REQUIRED = ["click", "pandas>=1.0.0", "statsmodels", "scipy"]
here = os.path.abspath(os.path.dirname(__file__))

# Loading the package's long description from a file in a standardized way.
def load_long_description(file_name="README.rst", encoding="utf-8"):
    """Loads the long description from the README file."""
    with io.open(os.path.join(here, file_name), encoding=encoding) as f:
        return "\n" + f.read()

long_description = load_long_description()

# Loading package version in a safer, more concise manner.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# Custom command for uploading the package, encapsulated in a class.
class UploadCommand(Command):
    """Support setup.py upload."""
    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{}\033[0m".format(s))

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
        os.system(f"{sys.executable} setup.py sdist bdist_wheel --universal")

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        sys.exit()

# Setup function call: structured to enhance readability and maintenance.
setup(
    name=NAME,
    version=about["__version__"],
    description=about["__description__"],
    long_description=long_description,
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    packages=find_packages(exclude=("tests",)),
    entry_points={"console_scripts": ["eemeter=eemeter.cli:cli"]},
    install_requires=REQUIRED,
    include_package_data=True,
    license=about["__license__"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    cmdclass={"upload": UploadCommand},
)
