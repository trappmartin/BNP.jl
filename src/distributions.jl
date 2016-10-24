export WishartGaussian, GaussianDiagonal, DirichletMultinomial, GammaNormal, NormalNormal, BetaBernoulli, ConjugatePostDistribution

abstract ConjugatePostDistribution

abstract UnivariateConjugatePostDistribution <: ConjugatePostDistribution
abstract DiscreteUnivariateConjugatePostDistribution <: UnivariateConjugatePostDistribution
abstract ContinuousUnivariateConjugatePostDistribution <: UnivariateConjugatePostDistribution

abstract MultivariateConjugatePostDistribution <: ConjugatePostDistribution
abstract DiscreteMultivariateConjugatePostDistribution <: MultivariateConjugatePostDistribution
abstract ContinuousMultivariateConjugatePostDistribution <: MultivariateConjugatePostDistribution

# Gaussian with Normal Inverse Wishart Prior
type WishartGaussian <: ContinuousMultivariateConjugatePostDistribution

    D::Int

    # sufficient statistics
    n::Int
    sums::Vector{Float64}
    ssums::Array{Float64}

    # base model parameters
    mu0::Vector{Float64}
    kappa0::Float64
    nu0::Float64
    Sigma0::Array{Float64}

    function WishartGaussian(mu::Vector{Float64}, kappa::Float64,
            nu::Float64, Sigma::Array{Float64})

        d = length(mu)

        new(d, 0, zeros(d), zeros(d, d), mu, kappa, nu, Sigma)
    end

    function WishartGaussian(D::Int, mu::Vector{Float64}, kappa::Float64,
            nu::Float64, Sigma::Array{Float64})

        new(D, 0, zeros(d), zeros(d, d), mu, kappa, nu, Sigma)
    end

    function WishartGaussian(D::Int,
            n::Int, sums::Vector{Float64}, ssums::Array{Float64},
            mu::Vector{Float64}, kappa::Float64,
            nu::Float64, Sigma::Array{Float64})

        new(D, n, sums, ssums, mu, kappa, nu, Sigma)
    end

end

Base.show(io::IO, d::WishartGaussian) =
    show_multline(io, d, [(:dim, d.D), (:μ0, d.mu0), (:Σ0, d.Sigma0), (:κ0, d.kappa0), (:ν0, d.nu0)])

# Normal with Gamma prior
type GammaNormal <: ContinuousUnivariateConjugatePostDistribution

	# sufficient statistics
	n::Int
	sums::Float64
  ssums::Float64

	# model parameters
	μ0::Float64
	λ0::Float64
	α0::Float64
	β0::Float64

	function GammaNormal(;μ = 0.0, λ = 1.0, α = 1.0, β = 1.0)
		new(0, 0.0, 0.0, μ, λ, α, β)
	end

end

Base.show(io::IO, d::GammaNormal) =
    show_multline(io, d, [(:μ0, d.μ0), (:λ0, d.λ0), (:α0, d.α0), (:β0, d.β0)])

# Normal with Normal prior
type NormalNormal <: ContinuousUnivariateConjugatePostDistribution

	# sufficient statistics
	n::Int
	sums::Float64
  ssums::Float64

	# model parameters
	μ0::Float64
	σ0::Float64

	function NormalNormal(;μ0 = 0.0, σ0 = 1.0)
		new(0, 0.0, 0.0, μ0, σ0)
	end

end

Base.show(io::IO, d::NormalNormal) =
    show_multline(io, d, [(:μ0, d.μ0), (:σ0, d.σ0)])

# Gaussian with Diagonal Covariance
type GaussianDiagonal{T <: ContinuousUnivariateConjugatePostDistribution} <: ContinuousMultivariateConjugatePostDistribution

    # sufficient statistics
    dists::Vector{T}

    function GaussianDiagonal(dists::Vector{T})
        new(dists)
    end

end

# Multinomial with Dirichlet Prior
type DirichletMultinomial <: DiscreteMultivariateConjugatePostDistribution

    D::Int

    # sufficient statistics
    n::Int
    counts::SparseMatrixCSC{Int,Int}

    # base model parameters
    alpha0::Float64

  	# cache
  	dirty::Bool
  	Z2::Float64
  	Z3::Array{Float64}

    function DirichletMultinomial(D::Int, alpha::Float64)
        new(D, 0, sparsevec(zeros(D)), alpha, true, 0.0, Array{Float64}(0))
    end

    function DirichletMultinomial(N::Int, counts::Vector{Int},
                            D::Int, alpha::Float64)
        new(D, N, sparsevec(counts), alpha, true, 0.0, Array{Float64}(0))
    end

    function DirichletMultinomial(N::Int, counts::SparseMatrixCSC{Int,Int},
                            D::Int, alpha::Float64)
        new(D, N, counts, alpha, true, 0.0, Array{Float64}(0))
    end

end

# Bernoulli with Beta Prior
type BetaBernoulli <: DiscreteUnivariateConjugatePostDistribution

	# sufficient statistics
	successes::Int
  n::Int

	# beta distribution parameters
	α::Float64
	β::Float64

	function BetaBernoulli(;α = 1.0, β = 1.0)
		new(0, 0, α, β)
	end

end
