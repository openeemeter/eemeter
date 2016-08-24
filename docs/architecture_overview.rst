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
application (see :ref:`eemeter`).

Each of these components are open sourced under an MIT License and can be found
on github:

 - `eemeter <https://github.com/openeemeter/eemeter/>`_
 - `datastore <https://github.com/openeemeter/datastore/>`_
 - `etl <https://github.com/openeemeter/etl/>`_

The core calculation engine is separated from the datastore in order to allow
easier development of and evaluation of its methods, but this architecture
also makes it possible to embed the calculation engine or any of its useful
modules (such as the :ref:`weather module <eemeter-weather>`) in other
applications.

The data structures in each - the eemeter and the datastore - mirror each
other. This simplifies data transfer and eases interpretation of results.
