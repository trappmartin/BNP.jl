abstract ConjugatePostDistribution

# Gaussian with Normal Inverse Wishart Prior
type GaussianWishart <: ConjugatePostDistribution

    D::Int

    # sufficient statistics
    n::Int
    sums::Vector{Float64}
    ssums::Array{Float64}
    #TODO: use Float32, SSE -> simd

    # base model parameters
    mu0::Vector{Float64}
    kappa0::Float64
    nu0::Float64
    Sigma0::Array{Float64}

    function GaussianWishart(mu::Vector{Float64}, kappa::Float64,
            nu::Float64, Sigma::Array{Float64})

        d = length(mu)

        new(d, 0, zeros(d), zeros(d, d), mu, kappa, nu, Sigma)
    end

    function GaussianWishart(D::Int, mu::Vector{Float64}, kappa::Float64,
            nu::Float64, Sigma::Array{Float64})

        new(D, 0, zeros(d), zeros(d, d), mu, kappa, nu, Sigma)
    end

    function GaussianWishart(D::Int,
            n::Int, sums::Vector{Float64}, ssums::Array{Float64},
            mu::Vector{Float64}, kappa::Float64,
            nu::Float64, Sigma::Array{Float64})

        new(D, n, sums, ssums, mu, kappa, nu, Sigma)
    end

end

Base.show(io::IO, d::GaussianWishart) =
    show_multline(io, d, [(:dim, d.D), (:μ0, d.mu0), (:Σ0, d.Sigma0), (:κ0, d.kappa0), (:ν0, d.nu0)])

# Normal with Gamma
type NormalGamma <: ConjugatePostDistribution

	# sufficient statistics
	n::Int
	sums::Float64
    ssums::Float64

	# model parameters
	μ0::Float64
	λ0::Float64
	α0::Float64
	β0::Float64

	function NormalGamma(;μ = 0.0, λ = 1.0, α = 1.0, β = 1.0)
		new(0, 0.0, 0.0, μ, λ, α, β)
	end

end

Base.show(io::IO, d::NormalGamma) =
    show_multline(io, d, [(:μ0, d.μ0), (:λ0, d.λ0), (:α0, d.α0), (:β0, d.β0)])

# Multinomial with Dirichlet Prior
type MultinomialDirichlet <: ConjugatePostDistribution

    D::Int

    # sufficient statistics
    n::Int
    counts::SparseMatrixCSC{Int,Int}

    # base model parameters
    alpha0::Float64

	# cache
	dirty::Bool
	Z2::Float64
	Z3::Vector{Float64}

    function MultinomialDirichlet(D::Int, alpha::Float64)
        new(D, 0, sparsevec(zeros(D)), alpha, true, 0.0, Vector{Float64}(0))
    end

    function MultinomialDirichlet(N::Int, counts::Vector{Int},
                            D::Int, alpha::Float64)
        new(D, N, sparsevec(counts), alpha, true, 0.0, Vector{Float64}(0))
    end

    function MultinomialDirichlet(N::Int, counts::SparseMatrixCSC{Int,Int},
                            D::Int, alpha::Float64)
        new(D, N, counts, alpha, true, 0.0, Vector{Float64}(0))
    end

end

# Binomial with Beta Prior
type BinomialBeta <: ConjugatePostDistribution

    D::Int

	# sufficient statistics
	n::Int
	counts::Array{Int}

	# beta distribution parameters
	α::Float64
	β::Float64

	function BinomialBeta(D::Int;α = 1.0, β = 1.0)
		new(D, 0, zeros(D), α, β)
	end

end
