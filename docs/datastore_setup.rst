Development Setup
-----------------

Clone the repo and change directories
"""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    git clone git@github.com:impactlab/oeem-energy-datastore.git
    cd oeem-energy-datastore

Install required python packages
""""""""""""""""""""""""""""""""

We recommend using virtualenv (or virtualenvwrapper) to manage python packages

.. code-block:: bash

    mkvirtualenv oeem-energy-datastore
    pip install -r requirements.txt
    pip install -r dev-requirements.txt

Define the necessary environment variables
""""""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    # django
    export DJANGO_SETTINGS_MODULE=oeem_energy_datastore.settings
    export SECRET_KEY=<django-secret-key>  # random string

    # postgres
    export DATABASE_URL=postgres://user:password@host:5432/dbname

    # for API docs - should reflect the IP or DNS name where datastore will be deployed
    export SERVER_NAME=0.0.0.0:8000
    export PROTOCOL=http  # or https

    # For development only
    export DEBUG=true

    # For celery background tasks
    export CELERY_ALWAYS_EAGER=true

      or

    export BROKER_TRANSPORT=redis
    export BROKER_URL=redis://user:password@host:9549

If developing on the datastore, you might consider adding these to your
virtualenv postactivate script:

.. code-block:: bash

    vim /path/to/virtualenvs/oeem-energy-datastore/bin/postactivate

    # Refresh environment
    workon oeem-energy-datastore

Run database migrations
"""""""""""""""""""""""

.. code-block:: python

    python manage.py migrate

Seed the database
"""""""""""""""""

.. code-block:: python

    python manage.py dev_seed

Start a development server
""""""""""""""""""""""""""

.. code-block:: python

    python manage.py runserver
