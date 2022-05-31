Development
===========

Clone the project
-----------------

.. code-block:: bash

   git clone git@edgitlab.eurodecision.com:data/pandas-cleaner.git

Virtual environment
-------------------

Create a virtual environment

.. code-block:: bash

   python3 -m venv venv

Activate the created virtual environment

.. code-block:: bash

   . venv/bin/activate

On windows

.. code-block:: bash

   source venv/scripts/activate``

Install dependencies
--------------------

Pandas-cleaner depends on a few python packages, mainly pandas.
The list of the dependencies is provided in the requirements.txt file.
These dependencies can be easily installed using pip :

.. code-block:: bash

   pip install -r requirements.txt

Unit tests
----------

If you downloaded the source code and you have installed pytest, you can run the unit tests using the command:

.. code-block:: bash

   python3 -m pytest pdcleaner/tests

