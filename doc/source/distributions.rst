Distributions
===============

The following distributions are currently supported. We will add additional support for the Distributions package in near future.

Common Interface
-----------

A common interface to access the sufficient statistics and the log likelihood is provided for all distributions.

.. code-block:: julia

    julia> add_data!(dist, X) # add datum to dist
    julia> dist2 = add_data(dist, X) # add datum to copy of dist

.. code-block:: julia

    julia> remove_data!(dist, X) # remove datum from dist
    julia> dist2 = remove_data(dist, X) # remove datum from copy of dist

.. code-block:: julia

    julia> logpred(dist, X) # log likelihood datum under dist

Beta-Binomial
-----------

The Binomial distribution with Beta prior of dimensionality D can be created using:

.. code-block:: julia

    julia> dist = BinomialBeta(D) # with default α = 1.0 and β = 1.0
    julia> dist = BinomialBeta(D, α = 3, β = 4) # specify α and β parameter of Beta distribution


Dirichlet-Multinomial
-----------

The Multinomial distribution with Dirichlet prior of dimensionality D can be created using:

.. code-block:: julia

    julia> dist = MultinomialDirichlet(D, 1.0) # with default α = 1.0


Wishart-Gaussian
-----------

The Gaussian distribution with Wishart prior of dimensionality D can be created using:

.. code-block:: julia

    julia> dist = GaussianWishart(μ, κ, ν, Ψ) # with specified μ of dimensionality D, κ, ν and Ψ of dimensionality D x D
