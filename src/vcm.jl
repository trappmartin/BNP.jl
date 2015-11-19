"Variable Clustering Model Hyperparameters"
immutable VCMHyperparam <: AbstractHyperparam

  μ_x::Float64
  σ_x::Float64

  μ_g::Float64
  σ_g::Float64

  μ_noise::Float64
  σ_noise::Float64

  γ_noise_a::Float64
  γ_noise_b::Float64

  γ_g_a::Float64
  γ_g_b::Float64

  γ_alpha_a::Float64
  γ_alpha_b::Float64

  VCMHyperparam() = new(0.0, 1.0, 0.0, 1.0, 0.0, 0.1, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0)

end

"Variable Clustering Model Data Object"
type VCMData <: AbstractModelData

    # energy
    energy::Float64

    # latent factor matrix
    X::Array{Float64}

    # latent factor loadings
    G::Array{Float64}

    # assignment Matrix
    C::SparseMatrixCSC

    # number of clusters
    K::Int

    # concentration parameter (Dirichlet)
    α::Float64

    # sigma of Gaussian noise
    σ_noise::Float64

    # sigma of Gaussian prior on factor loadings
    σ_g::Float64

end

type VCMBuffer <: AbstractModelBuffer

  μ_x::Float64
  σ_x::Float64

  μ_g::Float64
  σ_g::Float64

  μ_noise::Float64
  σ_noise::Float64

  # Data Set
  Y::Array{Float64}

  # Dimesnionality
  D::Int

  # Number of Samples
  N::Int

  # latent factor matrix
  X::Array{Float64}

  # latent factor loadings
  G::Array{Float64}

  # assignment Matrix
  C::SparseMatrixCSC

  # assignments
  cc::Vector{Int}

  # number of clusters
  K::Int

  # concentration parameter (Dirichlet)
  α::Float64

  # number of aux. variables
  m_aux::Int

end

"Incremental initialisation as published by Palla et al."
function init_incremental_vcm(Y::Array{Float64}, hyper::VCMHyperparam; α = 1.0)

  (D, N) = size(Y)

  X = randn(1,N)
  G = randn(1,1)
  C = sparse(collect(1))
  cc = [1]

  B = VCMBuffer(
    hyper.μ_x,
    hyper.σ_x,
    hyper.μ_g,
    hyper.σ_g,
    hyper.μ_noise,
    hyper.σ_noise,
    Y,
    D,
    N,
    X,
    G,
    C,
    cc,
    1,
    α,
    3 )

  Yd = B.Y[1,:]

  sample_latent_factors!(Yd, B)
  B.Y = Yd
  gibbs_aux_assignment!(B)


  B.α = random_concentration_parameter(B.α, hyper.γ_alpha_a, hyper.γ_alpha_b, B.N, B.K)
  B.σ_noise = random_sigma_noise(B, hyper.γ_noise_a, hyper.γ_noise_b)
  B.σ_g = random_sigma_g(B, hyper.γ_g_a, hyper.γ_g_b)

  # loop over D-1 dimensions
  for d = 2:B.D

      Yd = cat(1, Yd, Y[d,:])

      # sample unobserved data from model
      B.Y = Yd
      gibbs_aux_assignment!(B, out_dim = d)
      # not sure why this is required...

      # run one iteration:
      sample_latent_factors!(Yd, B)
      gibbs_aux_assignment!(B)

      B.α = random_concentration_parameter(B.α, hyper.γ_alpha_a, hyper.γ_alpha_b, B.N, B.K)
      B.σ_noise = random_sigma_noise(B, hyper.γ_noise_a, hyper.γ_noise_b)
      B.σ_g = random_sigma_g(B, hyper.γ_g_a, hyper.γ_g_b)

  end

  B.Y = Y

  return B

end

"Train VCM using Gibbs, currently very slow."
function train_gibbs_vcm(B::VCMBuffer, hyper::VCMHyperparam; α = 1.0, burnin = 10, thinout = 1, maxiter = 100, verbose = false)

  results = VCMData[]

  B.α = α

  for i in 1:burnin

    sample_latent_factors!(B.Y, B)
    gibbs_aux_assignment!(B)

    B.α = random_concentration_parameter(B.α, hyper.γ_alpha_a, hyper.γ_alpha_b, B.N, B.K)
    B.σ_noise = random_sigma_noise(B, hyper.γ_noise_a, hyper.γ_noise_b)
    B.σ_g = random_sigma_g(B, hyper.γ_g_a, hyper.γ_g_b)
  end

  # run gibbs sweeps
  for i in 1:maxiter

    # run thinning
    for j in 1:thinout
        sample_latent_factors!(B.Y, B)
        gibbs_aux_assignment!(B)

        B.α = random_concentration_parameter(B.α, hyper.γ_alpha_a, hyper.γ_alpha_b, B.N, B.K)
        B.σ_noise = random_sigma_noise(B, hyper.γ_noise_a, hyper.γ_noise_b)
        B.σ_g = random_sigma_g(B, hyper.γ_g_a, hyper.γ_g_b)
    end

    # compute energy
    E = B.Y - ( (B.G .* B.C) * B.X )
    llh = -B.D*B.N*(log(B.σ_noise) + 0.5*log(2*pi)) - 0.5*sum(E.^2) / (B.σ_noise^2)

    # simplified normal pdf
    prior_g = -0.5 * (1.0/B.σ_g.^2) * sum( (B.G-repmat((B.μ_g * ones(B.D, 1)), 1, B.K)).^2 )
    prior_x = -0.5 * (1.0/B.σ_x.^2) * sum( (B.X-repmat((B.μ_x * ones(B.K, 1)), 1, B.N)).^2 )

    # dirichlet?
    Ns = map(c -> sum(B.cc .== c), unique(B.cc))
    prior_c = lgamma(B.α)+B.K*log(B.α) +sum(lgamma(Ns)) - lgamma(B.α+B.D)

    # energy
    e = llh + prior_g + prior_x + prior_c

    # store result of current Gibbs sweep
    push!(results, VCMData(e, deepcopy(B.X), deepcopy(B.G), deepcopy(B.C), B.K, B.α, B.σ_noise, B.σ_g))

    if verbose
      println("iteration: ", i, " with energy: ", e, " K: ", K)
    end
  end

  return results

end
