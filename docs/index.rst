.. warning::

   The `eemeter` package is under rapid development; we are working quickly
   toward a stable release. In the mean time, please proceed to use the package,
   but as you do so, recognize that the API is in flux and the docs might not
   be up-to-date. Feel free to contribute changes or open issues on
   `github <https://github.com/openeemeter/eemeter>`_ to report bugs, request
   features, or make suggestions.


The Open Energy Efficiency Meter
================================

This package holds the core methods used by the of the **Open Energy
Efficiency** energy efficiency metering stack. Specifically, the :code:`eemeter`
package abstracts the process of building and evaluating models of energy
consumption or generation and of using those to evaluate the effect of energy
efficiency interventions at a particular site associated with a particular
project.

The :code:`eemeter` package is only one part of the larger Open Energy Efficiency
technology stack. Briefly, the architecture of the stack is as follows:

  - :code:`eemeter`: Given project and energy data, the :code:`eemeter` package is
    responsible for creating models of energy usage under different project
    conditions, and for using those models to evaluate energy efficiency
    projects.
  - :code:`datastore`: The :code:`datastore` application is responsible for validating and
    storing project data and associated energy data, for using the :code:`eemeter` to
    evaluate the effectiveness of these projects using the data it stores, and
    for storing and serving those results. It exposes as REST API for
    handling these functions.
  - :code:`etl`: The :code:`etl` package provides tooling which helps to extract data from
    various formats, transform that data into the format accepted by datastore,
    and load that transformed data into the appropriate :code:`datastore` instance.
    ETL stands for Extract, Transform, Load.


Usage
-----

.. toctree::
   :maxdepth: 2

   guides

.. toctree::
   :maxdepth: 4

   eemeter_index

.. toctree::
   :maxdepth: 4

   datastore_index

.. toctree::
   :maxdepth: 4

   etl_index

References
----------

- `PRISM <http://www.marean.mycpanel.princeton.edu/~marean/images/prism_intro.pdf>`_
- `ANSI/BPI-2400-S-2012 <http://www.bpi.org/Web%20Download/BPI%20Standards/BPI-2400-S-2012_Standard_Practice_for_Standardized_Qualification_of_Whole-House%20Energy%20Savings_9-28-12_sg.pdf>`_
- `NREL's Uniform Methods <http://energy.gov/sites/prod/files/2013/11/f5/53827-8.pdf>`_

Licence
-------

MIT
