.. _datastore-running-meters:

Running meters
--------------

This topic page covers scheduling and executing meter runs on the datastore.

Background
~~~~~~~~~~

Running a meter means pulling trace data, matching it with
relevant project data, and evaluating its energy effiency performance. This is
the central task performed by the datastore, so if the specifics are
unfamiliar, there is a bit more background information worthy of review in the
:ref:`Methods Overview <methods-overview>` section of the :ref:`guides <guides>`.

.. note::

    We will use the :code:`requests` python package for making requests, but
    you could just as easily use a tool like cURL or Postman.

    If you have the `eemeter` package installed, you will also have the
    `requests` package installed, but if not, you can install it with:

    .. code-block:: bash

        $ pip install requests

    A request using the requests library looks like this:

    .. code-block:: python

        import requests
        url = "https://example.com"
        data = {
            "first_name": "John",
            "last_name": "Doe"
        }
        requests.post(url + "/api/users/", json=data)

    which is equivalent to::

        POST /api/users/ HTTP/1.1
        Host: example.com
        {
            "first_name": "John",
            "last_name": "Doe"
        }

Setup
~~~~~

For this demonstration, we will assume that you have the following setup,
although of course yours will likely differ:

    1. a datastore application running at :code:`https://example.openeemeter.org/`
    2. a project with primary key 1, associated with traces 2, 3
    3. a project with primary key 2, associated with trace 4
    4. a trace primary key 2 (:code:`ELECTRICITY_CONSUMPTION_SUPPLIED`)
    5. a trace primary key 3 (:code:`NATURAL_GAS_CONSUMPTION_SUPPLIED`)
    6. a trace primary key 4 (:code:`NATURAL_GAS_CONSUMPTION_SUPPLIED`)

You should run something like the following, which sets up the variables we
will be using below.

.. code-block:: python

    # setup
    import requests

    url = "https://example.openeemeter.org"
    access_token = "INSERT_TOKEN_HERE"
    headers = {"Authorization": "Bearer {}".format(access_token)}

Scheduling a single meter run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a few ways to schedule a meter run. The simplest is the following,
which triggers a meter run with default model and formatter for the specified
trace (primary key 2):

.. code-block:: python

    data = {"trace": 2}
    response = requests.post(url + "/api/v1/meter_runs/",
                             json=data, headers=headers)

.. code-block:: python

    >>> response.json()
    {
        'id': 1,
        'project': 1,
        'trace': 2,
        'status': 'PENDING',
        'meter_input': None,
        'formatter_class': None,
        'formatter_kwargs': None,
        'model_class': None,
        'model_kwargs': None,
        'failure_message': None,
        'traceback': None,
        'added': '2016-09-28T23:57:21.454235Z',
        'updated': '2016-09-28T23:57:21.454260Z'
    }

The response shows us the complete specification of the meter run behavior,
which is as follows:

    1. the project was determined implicitly from the trace,
    2. the status is :code:`"PENDING"`, which means the tasks is scheduled but
       not yet running or completed
    3. the :code:`meter_input` has not yet been created (this is the
       complete serialized input to the meter, as required by the eemeter.)
    4. the model class, formatter class, and keyword arguments are left blank,
       indicating that default values will be used.
    5. the failure message and traceback are unpopulated, indicating no errors
       in execution (yet)

If you wish, you can also specify many of these properties explicitly:

.. code-block:: python

    data = {
        "trace": 2,
        "project": 2,
        "model_class": "MyModel",
        "model_kwargs": {
            "parameter_1": 1.5,
            "parameter_2": [0.8, 0.2],
        },
        "formatter_class": "MyFormatter",
        "formatter_kwargs": {},
    }
    response = requests.post(url + "/api/v1/meter_runs/",
                             json=data, headers=headers)

.. code-block:: python

    >>> response.json()
    {
        'id': 2,
        'project': 2,
        'trace': 2,
        'status': 'PENDING',
        'meter_input': None,
        'model_class': 'MyModel',
        'model_kwargs': {
            'parameter_1': 1.5,
            'parameter_2': [0.8, 0.2],
        },
        'formatter_class': 'MyFormatter',
        'formatter_kwargs': {},
        'failure_message': None,
        'traceback': None,
        'added': '2016-09-28T23:58:35.233478Z',
        'updated': '2016-09-28T23:58:35.233492Z'
    }

Or, if you leave out the project and trace attributes, you can specify the
exact serialized input:

.. code-block:: python

    data = {
        "meter_input": {...},
    }
    response = requests.post(url + "/api/v1/meter_runs/",
                             json=data, headers=headers)

.. code-block:: python

    >>> response.json()
    {
        'id': 3,
        'project': None,
        'trace': None,
        'status': 'PENDING',
        'meter_input': 'https://example.storage.googleapis.com/media/meter_inputs/010f59ae-15e9-4c43-8431-d90f74504770.json',
        'formatter_class': None,
        'formatter_kwargs': None,
        'model_class': None,
        'model_kwargs': None,
        'failure_message': None,
        'traceback': None,
        'added': '2016-09-28T23:59:02.667663Z',
        'updated': '2016-09-28T23:59:02.667681Z'
    }


Scheduling bulk meter runs
~~~~~~~~~~~~~~~~~~~~~~~~~~

To schedule bulk meter runs, instead of specifying a project and/or trace, you
specify a set of targets, which are sets of project and/or trace.:

.. code-block:: python

    data = {
        "targets": [
            {
                "project": 1,
            },
            {
                "project": 2,
            }
        ]
    }
    response = requests.post(url + "/api/v1/meter_runs/bulk/",  # note: different url!
                             json=data, headers=headers)

.. code-block:: python

    >>> response.json()
    [
        [
            {
                'id': 4,
                'project': 1,
                'trace': 2,
                'status': 'PENDING',
                'meter_input': None,
                'formatter_class': None,
                'formatter_kwargs': None,
                'model_class': None,
                'model_kwargs': None,
                'failure_message': None,
                'traceback': None,
                'added': '2016-09-29T00:01:43.152522Z',
                'updated': '2016-09-29T00:01:43.152545Z'
            },
            {
                'id': 5,
                'project': 1,
                'trace': 3,
                'status': 'PENDING',
                'meter_input': None,
                'formatter_class': None,
                'formatter_kwargs': None,
                'model_class': None,
                'model_kwargs': None,
                'failure_message': None,
                'traceback': None,
                'added': '2016-09-29T00:01:43.152557Z',
                'updated': '2016-09-29T00:01:43.152576Z'
            }
        ],
        [
            {
                'id': 6,
                'project': 2,
                'trace': 4,
                'status': 'PENDING',
                'meter_input': None,
                'formatter_class': None,
                'formatter_kwargs': None,
                'model_class': None,
                'model_kwargs': None,
                'failure_message': None,
                'traceback': None,
                'added': '2016-09-29T00:01:43.152578Z',
                'updated': '2016-09-29T00:01:43.152590Z'
            }
        ]
    ]

Note how results are returned grouped by target; each of the traces associated
with the specified project are triggered simultaneously.

If model or formatter class or kwarg arguments are supplied, they will be
applied to all meter_runs.

Once you have completed meter runs, you can create aggreations of the results.

See how to run aggregations: :ref:`Running Aggregations <datastore-running-aggregations>`.
