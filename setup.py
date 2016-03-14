from setuptools import setup, find_packages, Command

version = __import__('eemeter').get_version()

long_description = "Standard methods for calculating energy efficiency savings."

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py', '--runslow', '--cov-report', 'term-missing', '--cov', 'eemeter'  ])
        raise SystemExit(errno)


setup(name='eemeter',
    version=version,
    description='Open Energy Efficiency Meter',
    long_description=long_description,
    url='https://github.com/impactlab/eemeter/',
    author='Matt Gee, Phil Ngo, Eric Potash',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    cmdclass = {'test': PyTest},
    keywords='open energy efficiency meter method methods calculation savings',
    packages=find_packages(),
    install_requires=['pint',
                      'pyyaml',
                      'scipy',
                      'numpy',
                      'pandas >= 0.18',
                      'requests',
                      'pytz'],
    package_data={'': ['*.json','*.gz']},
)
