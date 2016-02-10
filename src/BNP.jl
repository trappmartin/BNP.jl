VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module BNP

    using Distributions
    using Clustering
    using ArrayViews
    using StatsBase

    # source files
    include("distributions.jl")
    include("distfunctions.jl")

		include("show.jl")
    include("utils.jl")
    include("common.jl")

    include("datasets.jl")

    include("dpmm.jl") # Dirichlet Process Mixture Model

    include("vcm_utils.jl")
    include("vcm.jl") # Variable Clustering Model

    include("hdp_utils.jl")
    include("hdp.jl") # Hierarchical Dirichlet Proces

    # analysis functions
    include("psm.jl")
    include("vipointestimate.jl")


    # define models
    abstract ModelType

    type DPM <: ModelType
      H::ConjugatePostDistribution
      α::Float64

      DPM(H::ConjugatePostDistribution; α = 1.0) = new(H, α)
    end

    type VCM <: ModelType
      α::Float64

      VCM(;α = 1.0) = new(α)
    end

    type HDP <: ModelType
      H::ConjugatePostDistribution
      α::Float64
      γ::Float64

      HDP(H::ConjugatePostDistribution; α = 0.1, γ = 5.0) = new(H, α, γ)
    end

    abstract SamplingType
    type Gibbs <: SamplingType
        burnin::Int
        thinout::Int
        maxiter::Int

        Gibbs(; burnin = 0, thinout = 1, maxiter = 100) = new(burnin, thinout, maxiter)

    end

    abstract InitialisationType

    type PrecomputedInitialisation <: InitialisationType
      Z::Array{Int}
      PrecomputedInitialisation(Z::Array{Int}) = new(Z)
    end

    type RandomInitialisation <: InitialisationType
      k::Int

      RandomInitialisation(;k = 2) = new(k)
    end

    type KMeansInitialisation <: InitialisationType
      k::Int

      KMeansInitialisation(;k = 2) = new(k)
    end

    type IncrementalInitialisation <: InitialisationType end

    # define helper functions
    function train(model::DPM, sampler::Gibbs, init::RandomInitialisation, X::AbstractArray)

      # init
      (Z, G) = init_random_dpmm(X, model.H, k = init.k)

      # inference
      return train_cgibbs_dpmm(X, model.H, Z, G, DPMHyperparam(), alpha = model.α, burnin = sampler.burnin, thinout = sampler.thinout, maxiter = sampler.maxiter)

    end

    function train(model::DPM, sampler::Gibbs, init::PrecomputedInitialisation, X::AbstractArray)

      # init
      (Z, G) = init_precomputed_dpmm(X, model.H, init.Z)

      # inference
      return train_cgibbs_dpmm(X, model.H, Z, G, DPMHyperparam(), alpha = model.α, burnin = sampler.burnin, thinout = sampler.thinout, maxiter = sampler.maxiter)

    end

    function train(model::DPM, sampler::Gibbs, init::KMeansInitialisation, X::AbstractArray)

      # init
      (Z, G) = init_kmeans_dpmm(X, model.H, k = init.k)

      # inference
      return train_cgibbs_dpmm(X, model.H, Z, G, DPMHyperparam(), alpha = model.α, burnin = sampler.burnin, thinout = sampler.thinout, maxiter = sampler.maxiter)

    end

    function train(model::VCM, sampler::Gibbs, init::IncrementalInitialisation, X::AbstractArray)

      # init
      B = init_incremental_vcm(X, VCMHyperparam(); α = model.α)

      # inference
      return train_gibbs_vcm(B, VCMHyperparam(), α = model.α, burnin = sampler.burnin, thinout = sampler.thinout, maxiter = sampler.maxiter)

    end

    function train{T <: Real}(model::HDP, sampler::Gibbs, init::RandomInitialisation, X::Vector{Vector{T}})

      # init
      (Z, G) = init_random_hdp(X, model.H, k = init.k)

      # inference

      return train_gibbs_hdp(X, model.H, Z, G, HDPHyperparam(), init.k, α = model.α, γ = model.γ, maxiter = sampler.maxiter)

    end

    # export commands
    export
      # Models
      DPM,
      VCM,
      HDP,

      # Inference Algorithms
      Gibbs,

      # Initialisations
      PrecomputedInitialisation,
      IncrementalInitialisation,
      RandomInitialisation,
      KMeansInitialisation,

      # Parameters
      DPMHyperparam,
      VCMHyperparam,
      HDPHyperparam,

      # Resulting Models
      DPMData,
      VCMData,
      HDPData,

      # Distributions
      ConjugatePostDistribution,
      GaussianWishart,
	  NormalGamma,
      MultinomialDirichlet,
	  BinomialBeta,

      # training method
      train,

      # expose Distribution functions
      add_data!,
      remove_data!,
      logpred,

      # expose utility functions
      rand_indices,

      # dataset generation
      generateBarsDataset,

	  # analysis
	  compute_psm,
      point_estimate

end
