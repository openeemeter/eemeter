.. _architecture-overview:

Architecture Overview
---------------------

The complete eemeter architecture consists primarily of a datastore
application (see :ref:`datastore`), which houses energy and project data, and
a data pipeline toolkit (see :ref:`etl-toolkit`) that helps get data into the
datastore.

These two work in tandem to take raw energy data in whatever form it exists
and compute energy savings using the eemeter package. The methods and models
used within the datastore for computing energy savings are kept in a library
package called eemeter, which can also be used independent of the datastore
application (see `eemeter`).


Each of these components are open sourced under an MIT License and can be found
on github:

 - `**etl** <https://github.com/impactlab/oeem-etl/>`_
 - `**eemeter** <https://github.com/impactlab/eemeter/>`_
 - `**datastore** <https://github.com/impactlab/oeem-energy-datastore/>`_
