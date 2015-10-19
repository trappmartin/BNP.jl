using Distributions

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

# Multinomial with Dirichlet Prior
type MultinomialDirichlet <: ConjugatePostDistribution
    
    D::Int
    
    # sufficient statistics
    n::Int
    counts::SparseMatrixCSC{Int,Int}
    
    # base model parameters
    alpha0::Float64
    #p0::Vector{Float64}
    
    function MultinomialDirichlet(D::Int, alpha::Float64)
        new(D, 0, sparsevec(zeros(D)), alpha)
    end
    
    function MultinomialDirichlet(N::Int, counts::Vector{Int}, 
                            D::Int, alpha::Float64)
        new(D, N, sparsevec(counts), alpha)
    end
    
    function MultinomialDirichlet(N::Int, counts::SparseMatrixCSC{Int,Int}, 
                            D::Int, alpha::Float64)
        new(D, N, counts, alpha)
    end
    
end