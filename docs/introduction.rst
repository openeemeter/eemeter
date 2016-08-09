.. _introduction:

Introduction
------------

The OpenEEmeter is an open source software package that uses metered energy
data to manage aggregate demand capacity across a portfolio of retail
customer accounts. The software package consists of three main parts:

1. an Extract-Transform-Load (ETL) toolkit for processing project,
   energy, and building data
   (`github <https://github.com/impactlab/oeem-etl/>`_);
2. a core calculation library (this package) that implements standardized
   methods (`github <https://github.com/impactlab/eemeter/>`_); and
3. a datastore application for storing post-ETL inputs and computed outputs
   (`github <https://github.com/impactlab/oeem-energy-datastore/>`_).

More information about this architecture can be found in
:ref:`architecture-overview`.

Core use cases
^^^^^^^^^^^^^^

The OpenEEmeter has been designed specifically to provide weather-normalized
energy savings measurements for a portfolio of projects using monthly billing
data or interval smart meter data. The main outputs for this core use case
are project and portfolio-level are:

- Gross Energy Savings
- Annualized Energy Savings
- Realization Rate (when savings predictions are available)

More information about these methods can be found in
:ref:`methods-overview`.

Other potential use cases
^^^^^^^^^^^^^^^^^^^^^^^^^

The OpenEEmeter can also be configured to manage energy resources across a
portfolio of buildings, including potentially:

- Analytics of raw energy data
- Portfolio management
- Demand side resource management

Data requirements
^^^^^^^^^^^^^^^^^

The EEmeter requires a combination of trace data, project data, and
weather data to calculate weather-normalized savings. At its most rudimentary,
the EEmeter requires a :ref:`trace <glossary-trace>` of consumption data
along with project data indicating the completion date and location of the
project.

The completion of a :ref:`project <glossary-project>` demarcates the shift
between a :ref:`baseline modeling period <glossary-baseline-period>` and a
:ref:`reporting modeling period <glossary-reporting-period>`. For more
information on this, see :ref:`methods-overview`.

The EEmeter is configured to manage :ref:`project <glossary-project>` and
:ref:`trace <glossary-trace>` data. Trace data can be electricity, natural gas,
or solar photovoltaic data of any frequency - from monthly billing data to
high-frequency sensor data (see :ref:`meters-and-smart-meters`).

Where project and trace data originate from different database sources, a
common key must be available to link projects with their respective traces.

Project data
""""""""""""

Project data is typically a set of attributes that can be used for advanced
savings analytics, but at minimum must contain a date to demarcate start and
end of :ref:`intervention <glossary-intervention>` periods.

Each project must have, at minimum:

- a unique project id
- start and end dates of known interventions
- a ZIP code (for gathering associated weather data)
- a set of associated traces

Other data can also be associated with projects, including (but not limited
to):

- savings predictions
- square footage
- cost

Trace data
""""""""""

Each trace must have, at minimum,

- a link to a project id
- a unique id of its own
- an :ref:`interpretation <glossary-trace-interpretation>`
- a set of records

Each record within a trace must have:

- a time period (start and end dates)
- a value and assiciated units of
- a boolean "estimated" flag

The EEmeter will reject traces not meeting built-in data sufficiency requirements.

Loading data
^^^^^^^^^^^^

The :ref:`eemeter` python package is a calculation engine which is not
desigend for data storage. Instead, project and trace data are stored
in the :ref:`datastore` alongside outputs from the :ref:`eemeter`.

To load data into the datastore, EEMeter comes bundled with an
:ref:`etl-toolkit`. If you are deploying the open source software, you will
need to write or customize a parser to load your data into the ETL pipeline.
We rely on a python module called `luigi <https://luigi.readthedocs.io/>`_
to manage the bulk importation of data.

More on this :ref:`architecture <architecture-overview>`.

External analysis
^^^^^^^^^^^^^^^^^

You may decide that you want to use EEmeter results to analyze project data
that does not get parsed and uploaded into the :ref:`datastore`. We have made
it easy to export your EEmeter results through an API or through a web
interface. Other options include a direct database connection to a BI tool
like Tableau or Salesforce.
