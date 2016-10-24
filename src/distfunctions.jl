"""
	add_data(d::ConjugatePostDistribution, X)

Add data to Posterior Distribution.
"""
function add_data(d::ConjugatePostDistribution, X)
	dist = deepcopy(d)
	add_data!(dist, X)
	return dist
end

"""
	add_data!(d::ConjugatePostDistribution, X)

Add data to Posterior Distribution (inplace).
"""
function add_data!(d::MultivariateConjugatePostDistribution, X::AbstractMatrix)
    @simd for i in 1:size(X, 1)
			@inbounds add_data!(d, @view X[i,:])
		end
end

function add_data!(d::UnivariateConjugatePostDistribution, X::AbstractVector)
	@simd for i in 1:size(X, 1)
		@inbounds add_data!(d, X[i])
	end
end

function add_data!(d::WishartGaussian, X::AbstractVector)
	@assert length(X) == d.D

	d.n += 1
  d.sums += X
  d.ssums += X * X'
end

function add_data!(d::GaussianDiagonal, X::AbstractVector)
	for (dim, dist) in enumerate(d.dists)
		add_data!(dist, X[dim])
	end
end

function add_data!(d::ContinuousUnivariateConjugatePostDistribution, X::AbstractFloat)
	d.n += 1
	d.sums += X
	d.ssums += X^2
end

function add_data!(d::DirichletMultinomial, X::AbstractVector)

    # process samples
    Dmin = minimum(X)
    Dmax = maximum(X)

    if Dmax > d.D
        throw(ArgumentError("Value of X and is larger than Multinomial Distribution!"))
    end

    if Dmin < 1
        throw(ArgumentError("Value of X and is smaller than 1!"))
    end

    d.n += length(X)

    for x in X
        d.counts[x] += 1
    end

    d.dirty = true
end

function add_data!(d::DirichletMultinomial, X::SparseVector{Int, Int})
		@assert length(X) == d.D

    d.n += 1
    d.counts += X
    d.dirty = true
end

function add_data!(d::BetaBernoulli, X::Integer)
	d.successes += X
	d.n += 1
end

"""
	remove_data(d::ConjugatePostDistribution, X::AbstractArray)

Remove data from Posterior Distribution.
"""
function remove_data(d::ConjugatePostDistribution, X)
		dist = deepcopy(d)
		remove_data!(dist, X)
		return dist
end

"""
	remove_data!(d::ConjugatePostDistribution, X)

Remove data from Posterior Distribution (inplace).
"""
function remove_data!(d::MultivariateConjugatePostDistribution, X::AbstractMatrix)
    @simd for i in 1:size(X, 1)
			@inbounds remove_data!(d, @view X[i,:])
		end
end

function remove_data!(d::UnivariateConjugatePostDistribution, X::AbstractVector)
	@simd for i in 1:size(X, 1)
		@inbounds remove_data!(d, X[i])
	end
end

function remove_data!(d::WishartGaussian, X::AbstractVector)
		@assert length(X) == d.D

    d.n -= 1
    d.sums -= X
    d.ssums -= X * X'
end

function remove_data!(d::ContinuousUnivariateConjugatePostDistribution, X::AbstractFloat)
    d.n -= 1
    d.sums -= X
    d.ssums -= X^2
end

function remove_data!(d::GaussianDiagonal, X::AbstractVector)
	for (dim, dist) in enumerate(d.dists)
		remove_data!(dist, X[dim])
	end
end

function remove_data!(d::DirichletMultinomial, X::AbstractVector)

    # process samples
    Dmin = minimum(X)
    Dmax = maximum(X)

    if Dmax > d.D
        throw(ArgumentError("Value of X and is larger than Multinomial Distribution!"))
    end

    if Dmin < 1
        throw(ArgumentError("Value of X and is smaller than 1!"))
    end

    d.n -= length(X)

    for x in X
        d.counts[x] -= 1
    end

    d.dirty = true
end

function remove_data!(d::DirichletMultinomial, X::SparseVector{Int, Int})

   # process samples
   @assert length(X) == d.D

   d.n -= 1
   d.counts -= X
   d.dirty = true
end

function remove_data!(d::BetaBernoulli, X::Bool)
		d.successes -= X
		d.n -= 1
end

"""
	isdistempty(d::ConjugatePostDistribution)

Check if distribution is empty (contains no samples).
"""
function isdistempty(d::ConjugatePostDistribution)
    return d.n <= 0
end

"""
	logpred(d::ConjugatePostDistribution, X)

Compute log posterior predictive.
"""
function logpred(d::WishartGaussian, x)

	if isdistempty(d)
		# posterior predictive of Normal Inverse Wishart is student-t Distribution

		C = d.Sigma0 * ((d.kappa0 + 1) / (d.kappa0 * (d.nu0 - d.D + 1)))

		dist = Distributions.MvTDist(d.nu0 - d.D + 1, d.mu0, C)
	  return Distributions.logpdf(dist, x)

	end

  # statistics
  sample_mu = d.sums / d.n

  # make sure values are not NaN
  sample_mu[sample_mu .!= sample_mu] = 0

  # compute posterior parameters
  kappa = d.kappa0 + d.n
  nu = d.nu0 + d.n
  mu = (d.kappa0 * d.mu0 + d.n * sample_mu) / kappa
  Sigma = d.Sigma0 + d.ssums - kappa * (mu * mu') + ( d.kappa0 * (d.mu0 * d.mu0') )

  # posterior predictive of Normal Inverse Wishart is student-t Distribution, see:
  # K. Murphy, Conjugate Bayesian analysis of the Gaussian distribution. Eq. 258

 C = Sigma * ((kappa + 1) / (kappa * (nu - d.D + 1)))

 if !isposdef(C)
	 throw(ErrorException("C is not positive definite! (C: $(C))"))
 end

  dist = Distributions.MvTDist(nu - d.D + 1, mu, C)
  return Distributions.logpdf(dist, x)

end

"Log PDF of Generalized student-t Distribution."
function tlogpdf(x, df, mean, sigma)

   function tdist_consts(df, sigma)
       hdf = 0.5 * df
       shdfhdim = hdf + 0.5
       v = lgamma(hdf + 1/2) - lgamma(hdf) - 0.5*log(df) - 0.5*log(pi) - log(sigma)
       return (shdfhdim, v)
   end

    shdfhdim, v = tdist_consts(df, sigma)

    xx = x .- mean
    xx = (xx ./ sqrt(sigma)).^2

    p = 1/df * xx

    return v - log((1 + p).^((df+1) / 2))
end

"Log PDF for NormalGamma."
function logpred(d::GammaNormal, x)

   if d.n == 0
      # posterior predictive of Normal Gamma is student-t Distribution
      df = 2 * d.α0
      mean = d.μ0
      sigma = ( d.β0 * (d.λ0 + 1) ) / (d.λ0 * d.α0)

      return tlogpdf(x, df, mean, sigma)
   end


    # statistics
    sample_mu = d.sums / d.n

    # make sure values are not NaN
    sample_mu = sample_mu != sample_mu ? 0 : sample_mu

    # compute posterior parameters
    μ = (d.λ0 * d.μ0 + d.sums) / (d.λ0 + d.n)
    λ = d.λ0 + d.n
    α = d.α0 + (d.n / 2)
    s = (d.ssums / d.n) - (sample_mu * sample_mu)
    β = d.β0 + 1/2 * (d.n * s + (d.λ0 * d.n * (sample_mu - d.μ0)^2 ) / (d.λ0 + d.n) )

    # posterior predictive of Normal Gamma is student-t Distribution
    df = 2 * α
    mean = μ
    sigma = ( β * (λ + 1) ) / (λ * α)

		return tlogpdf(x, df, mean, sigma)
end

function logpred(d::GaussianDiagonal, x)
	return vec(sum([logpred(di, x[dim, :]) for (dim, di) in enumerate(d.dists)]))
end

"Log PDF for NormalNormal."
function logpred(d::NormalNormal, x)
	return Float64[logpred(d, xi) for xi in x]
end

function logpred(d::NormalNormal, x::Number)

   if d.n == 0
		 return normlogpdf(d.μ0, d.σ0, x)
   end

  # statistics
  sample_mu = d.sums / d.n
	sample_var = (d.n * d.ssums - d.sums^2) / (d.n*(d.n-1))
	sample_var = (d.ssums - (d.sums^2)/d.n) / (d.n - 1)
	sample_var = 10.0

	# make sure values are not NaN
	sample_mu = sample_mu != sample_mu ? 0 : sample_mu

	σ = (sample_var * d.σ0) / (d.n * d.σ0 + sample_var)
  μ = σ * (d.n*sample_mu/sample_var + d.μ0/d.σ0)

	return normlogpdf(μ, sqrt(σ), x)
end

"Log PMF for MultinomialDirichlet."
function logpred(d::DirichletMultinomial, x)

    N = length(x)

	 # TODO: This is bad and slow code, improve!!!
    # construct sparse vector
    xx = spzeros(d.D, N)
    for i in 1:N
      xx[x[i]] += 1
    end

    m = sum(xx)
    mi = nnz(xx)

	 if d.dirty
		 d.Z2 = lgamma(d.alpha0 + d.n)
		 d.Z3 = lgamma( d.alpha0/d.D + d.counts )

		 d.dirty = false
	 end

	l1 = lgamma(m + 1) - sum(lgamma(mi + 1))
	l2 = d.Z2 - lgamma(d.alpha0 + d.n + m)
	l3 = sum( lgamma( d.alpha0 / d.D + d.counts + xx ) - d.Z3 )

    return [l1 + l2 + l3]
end

"Log PMF for MultinomialDirichlet."
function logpred(d::DirichletMultinomial, x::SparseVector{Int, Int})

   D = size(x, 1)
   N = size(x, 2)

   if N > 1
      throw(ErrorException("Multiple samples are not supported yet!"))
   end

   m = sum(x)
   mi = nnz(x)

   if d.dirty
      d.Z2 = lgamma(d.alpha0 + d.n)
      d.Z3 = lgamma( d.alpha0/d.D + d.counts )

      d.dirty = false
   end

	l1 = lgamma(m + 1) - sum(lgamma(mi + 1))
	l2 = d.Z2 - lgamma(d.alpha0 + d.n + m)
	l3 = sum( lgamma( d.alpha0 / d.D + d.counts + x ) - d.Z3 )

   return [l1 + l2 + l3]
end

"Log PMF for BernoulliBeta."
function logpred(d::BetaBernoulli, X::AbstractArray)
	return Float64[logpred(d, x) for x in X]
end

function logpred(d::BetaBernoulli, X::Bool)

  	# posterior
		α = d.α + d.successes
		β = d.β + d.n - d.successes

   return log(α) - log(α + β)
end
