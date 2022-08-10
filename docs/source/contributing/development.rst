Developments 
============

.. note::

   Before starting new code, we highly recommend opening an issue on
   `GitHub <https://github.com/eurodecision/pandas-cleaner/issues>`_ to discuss potential changes.

This page is partly based on https://docs.github.com/en/get-started/quickstart/contributing-to-projects
Itis recommended to read it if it is your first time contributing to an open source project on github, 
or if you need more detailed explanations.

Fork the repo
-------------

1. Navigate to https://github.com/eurodecision/pandas-cleaner
2. Click ``Fork``
3. Available options are described on this page https://docs.github.com/en/get-started/quickstart/contributing-to-projects
   Default ones should be ok
4. Click on ``Create fork``


Clone the fork
--------------

1. On GitHub, navigate to your fork
2. Click on ``Code``
3. Clone the fork with your preferred method

Set up the development environment
----------------------------------

Navigate to your local version of the fork

.. code-block:: bash

   cd pandas-cleaner

Virtual environment
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

Pandas-cleaner depends on a few python packages, mainly pandas.
The list of the dependencies needed for devlopment is provided in the requirements.txt file.
These dependencies can be easily installed using pip :

.. code-block:: bash

   pip install -r requirements.txt

Install the package
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install -e .


Code guidelines
---------------

+ Please use `pep8 <https://pypi.python.org/pypi/pep8>`_ and `flake8 <http://flake8.pycqa.org/>`_ Python style guidelines.

  .. code-block:: bash

     pylint src
     flake8 src

+ Use `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ for docstrings. 

+ Changes must be accompanied by updated documentation and examples.

  .. code-block::

     make -C docs html

  You can see it by opening ``docs/build/html/index.html`` in your browser.
  
+ After making changes, ensure all tests pass and that your new code is covered by tests.
  You can run the unit tests using the command:

  .. code-block:: bash

     pytest

  and check the coverage

  .. code-block:: bash

     pytest --cov

  Optionnally, create a html report 

  .. code-block:: bash

     pytest --cov --cov-report html

  You can see it by opening ``htmlcov/index.html`` in your browser.

+ If your changes require new dependencies, check they are added in the ``requirements.txt`` and the ``setup.py``


Pushing changes
---------------

When you're ready, stage and commit your changes. 

.. code-block:: bash

   git add .
   git commit -m "a short description of the change"

and push the modifications to your forked project

.. code-block:: bash

   git push

Pull Request
------------

If you want to propose your changes into the main project.

1. On the page of your fork, click ``Contribute`` and then ``Open a pull request``.

GitHub will bring you to a page that shows the differences between your fork 
and the main repository. 

2. Click ``Create pull request``.

GitHub will bring you to a page where you can enter a title and a description of your changes.

.. note::

    It's important to provide as much useful information and a rationale for why you're making this pull request
    in the first place. The pandas-cleaner team needs to be able to determine whether your change is useful. 

3. Finally, click ``Create pull request``.

The pandas-cleaner team will review your contribution and provide a feedback.
If every requirement is valid, your changes will be merged in the devlopment branch and released
in the next release version.