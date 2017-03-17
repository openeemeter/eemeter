Management commands
-------------------

The following management commands are available for usage on the datastore.

`dev_seed`
^^^^^^^^^^

Creates an admin user:

  - **username**: :code:`demo`
  - **password**: :code:`demo-password`
  - **access token**: :code:`tokstr`

Creates a sample project with the id :code:`DEV_SEED_PROJECT` with the
following traces:

  - :code:`DEV_SEED_TRACE_NATURAL_GAS_MONTHLY`
  - :code:`DEV_SEED_TRACE_NATURAL_GAS_DAILY`
  - :code:`DEV_SEED_TRACE_ELECTRICITY_15MIN`
  - :code:`DEV_SEED_TRACE_ELECTRICITY_HOURLY`
  - :code:`DEV_SEED_TRACE_SOLAR_HOURLY`
  - :code:`DEV_SEED_TRACE_SOLAR_30MIN`

*Example usage*:

.. code-block:: bash

    python manage.py dev_seed

`prod_seed`
^^^^^^^^^^^

Creates an admin user with generated password and access token:

  - **username**: :code:`admin`
  - **password**: <generated password>
  - **access token**: <generated token>

The generated password and access token will be shown in the output:

.. code-block:: bash

    Admin password: <generated password>
    Admin token: <generated token>

*Example usage*:

.. code-block:: bash

    python manage.py prod_seed

`trace_record_indexes`
^^^^^^^^^^^^^^^^^^^^^^

Creates and destroy indexes as part of loading TraceRecords.

Loading raw data is significantly faster if indexes and foreign key
constraints are dropped and rebuilt after importing.

This command inspects the current indexes and constraints, dropping all but the
primary key indexes.

If new indexes are added, they should be added here (not in model classes) so
that they are properly rebuilt during imports.

The results of this command can be inspected through psql::

    => \d datastore_tracerecord

With indexes, the description will look something like this::

    Indexes:
        "datastore_tracerecord_pkey" PRIMARY KEY, btree (id)
        "datastore_tracerecord_ffe73c23" btree (trace_id)
    Foreign-key constraints:
        "datast_trace_id_53e4466e_fk_datastore_trace_id"
        FOREIGN KEY (trace_id) REFERENCES datastore_trace(id)
        DEFERRABLE INITIALLY DEFERRED

Without indexes, it will look something like this::

    Indexes:
        "datastore_tracerecord_pkey" PRIMARY KEY, btree (id)

*Example usage*:

To destroy trace_records (before ETL):

.. code-block:: bash

    python manage.py trace_record_indexes destroy

To create trace_records (after ETL):

.. code-block:: bash

    python manage.py trace_record_indexes create

`run_meters`
^^^^^^^^^^^^

Triggers meter runs for specified projects or traces.

*Example usage*:

.. code-block:: bash

    python manage.py run_meters --all-traces

*Optional arguments*::

      --projects PROJECTS [PROJECTS ...]
                            Project ids to run
      --traces TRACES [TRACES ...]
                            Trace ids to run
      --all-projects        Run meters for all projects, overrides --projects
      --all-traces          Run meters for all traces, overrides --traces
      --use-project-id      Use project_id, not id, for any projects to run
      --use-trace-id        Use trace_id, not id, for any traces to run
      --purge-queue         Purges celery queue before adding meter runs
      --detailed-output     Provides more detailed project and trace level output
                            re: meter ids
      --delete-previous-meters
                            Delete old meter runs associated with these ids

`meter_progress`
^^^^^^^^^^^^^^^^

Check progress of one or more meter runs.

*Example usage*:

.. code-block:: bash

    python manage.py meter_progress --all-meters

*Optional arguments*::

      --meters METERS [METERS ...]
                            Meter ids to check
      --all-meters          Check progress for all meters
      --poll-until-complete
                            Repeatedly check progress until all meters complete
      --poll-interval POLL_INTERVAL
                            Seconds to wait between checks if --poll-until-
                            complete
      --poll-max POLL_MAX   Max number of seconds to poll if --poll-until-complete
                            before exiting

`delete_meters`
^^^^^^^^^^^^^^^

Delete meter runs.

*Example usage*:

.. code-block:: bash

    python manage.py delete_meters

*Optional arguments*::

      --meters METERS [METERS ...]
                            Meter ids to delete
      --traces TRACES [TRACES ...]
                            Trace ids to delete associated meters
      --projects PROJECTS [PROJECTS ...]
                            Project ids to delete associated meters

`run_aggregations`
^^^^^^^^^^^^^^^^^^

Run aggregations of meter results by group.

*Example usage*:

.. code-block:: bash

    python manage.py run_aggregations --all-groups

*Optional arguments*::

      --group-names GROUP_NAMES [GROUP_NAMES ...]
                            Groups against which to run aggregations
      --all-groups          Run aggregations for all groups; overrides
                            --group_names

`meterresultmart`
^^^^^^^^^^^^^^^^^

Create and destroy the data warehouse mart for meter results.

The warehouse table is `warehouse_meterresultmart`

*Example usage*:

.. code-block:: bash

    python manage.py meterresultmart create
    python manage.py meterresultmart destroy

`modelresultmart`
^^^^^^^^^^^^^^^^^

Create and destroy the data warehouse mart for model results.

The warehouse table is `warehouse_modelresultmart`

*Example usage*:

.. code-block:: bash

    python manage.py modelresultmart create
    python manage.py modelresultmart destroy

`projectsummarymart`
^^^^^^^^^^^^^^^^^^^^

Create and destroy a data mart for metering results organized by project for
a charting frontend.

The warehouse table is `warehouse_projectsummarymart`

*Example usage*:

.. code-block:: bash

    python manage.py projectsummarymart create
    python manage.py projectsummarymart destroy

`tracesummarymart`
^^^^^^^^^^^^^^^^^^^^

Create and destroy a data mart that summarizes traces and their records.

The warehouse table is `warehouse_tracesummarymart`

*Example usage*:

.. code-block:: bash

    python manage.py tracesummarymart create
    python manage.py tracesummarymart destroy

`geoinfo`
^^^^^^^^^

Create and destroy two tables for geographical information

The warehouse tables are `warehouse_zctainfo` and `warehouse_countyinfo`

*Example usage*:

.. code-block:: bash

    python manage.py geoinfo create
    python manage.py geoinfo destroy
