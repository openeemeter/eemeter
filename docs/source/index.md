# EEmeter

<style>
.md-content .md-typeset h1 { display: none; }
</style>

<p align="center" id="openeemeter">
  <a href="#openeemeter"><img src="./images/openeemeter-logo-color.svg" alt="OpenEEmeter"></a>
</p>

<p align="center">
    <em>OpenEEmeter library, standardized framework for high performance energy prediction models.</em>
</p>

</p>
    <p align="center">
    <a href="https://github.com/openeemeter/eemeter" target="_blank">
        <img src="https://img.shields.io/github/license/openeemeter/eemeter.svg?logoColor=indigo" alt="License">
    </a>
    <a href="https://pypi.python.org/pypi/eemeter" target="_blank">
        <img src="https://img.shields.io/pypi/v/eemeter.svg?logoColor=indigo" alt="PyPi Version">
    </a>
    <a href="https://pypi.org/project/eemeter" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/eemeter.svg?logoColor=indigo" alt="Supported Python versions">
    </a>
    <a href="https://codecov.io/gh/openeemeter/eemeter" target="_blank">
        <img src="https://codecov.io/gh/openeemeter/eemeter/branch/master/graph/badge.svg" alt="Code Coverage">
    </a>
    <a href="https://github.com/ambv/black" target="_blank">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg?logoColor=indigo" alt="Code Coverage">
    </a>
</p>

---

**Source Code**: <a href="https://github.com/openeemeter/eemeter" target="_blank">https://github.com/openeemeter/eemeter</a>

---

EEmeter is an open source python library for creating standardized models for predicting energy usage. These standardized models are often used to calculate energy savings post demand side intervention (such as energy efficiency projects or demand response events). 

Key Features of EEmeter include:

- **Fast**: EEmeter models are extremely high performance, with model builds taking milliseconds. 
- **Open Source**: All code is open source, making EEmeter an excellent choice for measuring energy savings when results must be transparent and reproducible.
- **Multiple Model Types**: EEmeter offers both daily and hourly models, depending on the interval granularity available. The daily model is also compatible with billing data.
- **Easy and Intuitive**: The API interface is inspired by <a href="https://scikit-learn.org/stable/" target="_blank">scikit-learn</a>, a well-known data science package for building models. Just fit and predict.
- **Input Data Formatting**: Meter usage and temperature data is first routed through interval data classes to ensure standardization and avoid common pitfalls.
- **Data Sufficiency and Model Fit Checking**: Data sufficiency and model fit thresholds are built into the model, notifying users if limits are exceeded.
- **Model Serialization and Deserialization**: Models can be serialized into dictionaries or json objects and deserialized later.
- **Pandas DataFrame Support**: Input data and predictions use pandas DataFrames, a well-known format for data scientists and engineers.

