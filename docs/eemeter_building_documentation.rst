Building Documentation
----------------------


Documentation is built using the :code:`sphinx` package.
To build documentation, make sure that dev requirements are installed:

.. code-block:: bash

    $ pip install -r dev_requirements.txt

You will also need to `install pandoc <http://pandoc.org/installing.html>`_ to build docs locally.

And run the following from the root project directory.

.. code-block:: bash

    $ make -C docs html

To clean the build directory, run the following:

.. code-block:: bash

    $ make -C docs clean

