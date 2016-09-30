.. _datastore-running-aggregations:

Running aggregations
--------------------

We assume the same setup we used in :ref:`Running meters <datastore-running-meters>`

Scheduling a single aggregation run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregations of meter results are likewise scheduled through the API.
They are scheduled as unions of derivatives from 3 sets of objects: projects,
traces, or derivatives. You may specify any set of projects, traces, or
derivatives from which to draw derivatives for aggregation.

Since aggregations must be across like objects, trace interpretation and
derivative interpretation can be supplied as filters, or left implicit
(although you will get errors if there are inconsistencies).

The following will create an aggregation (sum) of derivatives from projects
1 and 2 with the interpretation annualized_weather_normal from traces matching
the interpretation "E_C_S".

.. code-block:: python

    data = {
        "projects": [1, 2],
        "derivatives": [],
        "traces": [],
        "trace_interpretation": "E_C_S",
        "derivative_interpretation": "annualized_weather_normal",
        "aggregation_interpretation": "SUM",
    }
    response = requests.post(url + "/api/v1/aggregation_runs/",
                             json=data, headers=headers)

.. code-block:: python

    >>> response.json()
    {
        'id': 1,
        'status': 'PENDING',
        'projects': [1, 2],
        'traces': [],
        'derivatives': [],
        'aggregation_input': 'https://example.storage.googleapis.com/media/aggregation_inputs/3cdfc090-ec80-4cc1-8faf-4ee8705393ab.json',
        'trace_interpretation': 'E_C_S',
        'derivative_interpretation': 'annualized_weather_normal',
        'aggregation_interpretation': 'SUM',
    }

Scheduling bulk aggregation runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mechanism for scheduling bulk aggregation runs is analogous to the mechanism
for scheduling bulk meter runs. If interpretation fields are left off, the
implication is that all types of aggregations should be attempted. Only
aggregations for which 1 or more derivative is available matching the
interpretation will be included. For the bulk method, interpretations should
be supplied as lists, as shown in comments.

.. code-block:: python

    data = {
        "targets": [
            {
                "projects": [1, 2],
                "derivatives": [],
                "traces": [],
                # 'trace_interpretations': ['E_C_S', 'NG_C_S'],
                # 'derivative_interpretations': ['annualized_weather_normal', 'gross_predicted'],
            }
        ]
    }
    response = requests.post(url + "/api/v1/aggregation_runs/bulk/",
                             json=data, headers=headers)

.. code-block:: python

    >>> response.json()
    [
        [
            {
                'id': 2,
                'status': 'PENDING',
                'derivatives': [],
                'projects': [1, 2],
                'traces': [],
                'aggregation_input': 'https://example.storage.googleapis.com/media/meter_inputs/010f59ae-15e9-4c43-8431-d90f74504770.json',
                'trace_interpretation': 'E_C_S',
                'derivative_interpretation': 'annualized_weather_normal',
                'aggregation_interpretation': 'SUM',
            },
            {
                'id': 3,
                'status': 'PENDING',
                'derivatives': [],
                'projects': [1, 2],
                'traces': [],
                'aggregation_input': 'https://example.storage.googleapis.com/media/meter_inputs/30eca307-93e5-4666-bb1f-4cf5be219c9b.json',
                'trace_interpretation': 'E_C_S',
                'derivative_interpretation': 'gross_predicted',
                'aggregation_interpretation': 'SUM',
            },
            {
                'id': 5,
                'status': u'PENDING',
                'derivatives': [],
                'projects': [1, 2],
                'traces': [],
                'aggregation_input': 'https://example.storage.googleapis.com/media/meter_inputs/7fc34cd6-e408-4a0d-bb3c-d504ae8f9357.json',
                'trace_interpretation': 'NG_C_S',
                'derivative_interpretation': 'annualized_weather_normal',
                'aggregation_interpretation': 'SUM',
            },
            {
                'id': 5,
                'status': u'PENDING',
                'derivatives': [],
                'projects': [1, 2],
                'traces': [],
                'aggregation_input': 'https://example.storage.googleapis.com/media/meter_inputs/c2d95844-4da1-475e-ae3d-d731dd5d3aa9.json',
                'trace_interpretation': 'NG_C_S',
                'derivative_interpretation': 'gross_predicted',
                'aggregation_interpretation': 'SUM',
            }
        ]
    ]
