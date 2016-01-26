.. _models

Models
===============

Common Interface
-----------

Each model can be trained using the common interface train:

.. code-block:: julia

	julia> models = train(DPM(G0), Gibbs(), KMeansInitialisation(), X)

Note that the train function allways returns a list of model objects. Those can objects can be used for further analysis. The fields of the model objects are described below.


Dirichlet Process Mixture Model
-----------
The Dirichlet Process Mixture Model (DP-MM) can be used to infer the number of clusters and their parameters. 

Trainin a DP-MM is done using

.. code-block:: julia

	julia> models = train(DPM(G0), Gibbs(), KMeansInitialisation(), X)
	
where G0 is the base distribution with conjugate prior.

The model object contains the following information:

.. code-block:: julia

	julia> # Energy
	julia> energy::Float64
	
	julia> # Dirichlet concentration parameter
	julia> α::Float64
	
	julia> # Distributions
	julia> distributions::Array{ConjugatePostDistribution}
	
	julia> # Data Point Assignments to Clusters
	julia> assignments::Array{Int}
	
	julia> # Weights
	julia> weights::Array{Float64}


Hierarcical Dirichlet Process Model
-----------

The Hierarcical Dirichlet Process Model (HDP) can be used to infer the number of topics (shared distributions) and their parameters. 

Trainin a HDP is done using

.. code-block:: julia

	julia> models = train(HDP(G0), Gibbs(), RandomInitialisation(), X)
	
where G0 is the base distribution with conjugate prior.

The model object contains the following information:

.. code-block:: julia

	julia> # Energy
	julia> energy::Float64
	
	julia> # Topics
	julia> distributions::Vector{ConjugatePostDistribution}
	
	julia> # Word Assignments per Document to Clusters
	julia> assignments::Vector{Vector{Int}}
	
	julia> # Topic Weights per Document
	julia> weights::Vector{Vector{Float64}}
	
Variable Cluster Model
-----------

