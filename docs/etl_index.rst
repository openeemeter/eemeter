.. _etl-toolkit:

ETL Toolkit
===========

The ETL toolkit is provided to assist moving data from its source into the
datastore.

"ETL" stands for Extract-Transform-Load. These three steps outline the
actions the ETL toolkit helps with and are as follows:

- **Extract**: obtain data from an external (non-datastore) source.

- **Transform**: convert that data into a form usable the datatore.

- **Load**: move the transformed data into the datastore.


The ETL library is not run directly. Rather, its components are used to build
ETL pipelines that are specific to a datastore instance.

.. toctree::
   :maxdepth: 4

   etl_installation


.. toctree::
   :maxdepth: 4

   etl_api
