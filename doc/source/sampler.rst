Getting Started
===============

Installation
-----------
The *BNP* package is currently not available through the Julia package system but can easily installed by running ``Pkg.clone("https://github.com/trappmartin/BNP.jl")``.

Clustering data using Dirichlet Process Mixture Model
-----------

In this example we start by drawing 100 observations from two bivariate Normal distributions.

.. code-block:: julia

    julia> X = cat(2, rand(2, 50), rand(2, 50) + 10)
    julia> Y = cat(2, zeros(50), ones(50))

Now we can initialize the package and construct a Gaussian data distribution using a Normal Inverse Wishart prior.

.. code-block:: julia
	
	julia> using BNP
	julia> μ0 = vec( mean(X, 2) )
	julia> κ0 = 1.0
	julia> ν0 = 4.0
	julia> Ψ = eye(2) * 10
	julia> G0 = GaussianWishart(μ0, κ0, ν0, Ψ)

After constructing G0 we can easily apply a Dirichlet Process Mixture Model using collapsed Gibbs sampling.

.. code-block:: julia

    julia> models = train(DPM(G0), Gibbs(), KMeansInitialisation(), X)
	
Please note that this example can also be found in the demos folder, allowing interactive exploration of the model.