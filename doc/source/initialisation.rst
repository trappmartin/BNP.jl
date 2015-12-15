.. _initialisation

Initialization Methods
===============

In order to initialize the Bayesian nonparametric models we provide a set of initialization approaches.
Currently not every initialization approach is available for all models.

Random Initialization
~~~~~~~~~~~~~~~~~~~~~~

The Random Initialization randomly assigns the data to a predefined number of groups.

.. code-block:: julia
    julia> init = RandomInitialisation() #  Random Initialization with k = 2
    julia> init = RandomInitialisation(k = 5) #  Random Initialization with k = 5

Incremental Initialization
~~~~~~~~~~~~~~~~~~~~~~

The Incremental Initialization sequentially assigns the data to groups.

.. code-block:: julia
    julia> init = IncrementalInitialisation() #  Incremental Initialization k = 5

K-Means Initialization
~~~~~~~~~~~~~~~~~~~~~~

The K-Means Initialization assigns the data using k-Means clustering to a predefined number of groups.

.. code-block:: julia
    julia> init = KMeansInitialisation() #  Random Initialisation with k = 2
    julia> init = KMeansInitialisation(k = 5) #  Random Initialisation with k = 5
