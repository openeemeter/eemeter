from setuptools import setup, find_packages

version = __import__('eemeter').get_version()


setup(
    name='eemeter',
    version=version,
    description='Open Energy Efficiency Meter',
    long_description=(
        "Standard methods for calculating energy efficiency savings."
    ),
    url='https://github.com/openeemeter/eemeter/',
    author='Open Energy Efficiency',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='open energy efficiency meter',
    packages=find_packages(),
    install_requires=[
        'holidays',
        'lxml',
        'numpy >= 1.10.2',
        'scipy',
        'pandas >= 0.19.2',
        'patsy',
        'pytz',
        'requests',
        'scikit-learn',
        'statsmodels >= 0.8.0rc1',
        'SQLAlchemy',
    ],
    package_data={'': ['*.json', '*.gz']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
