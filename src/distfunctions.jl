using ArrayViews

@doc doc"""
Convert ConjugatePostDistribution to Distribution.

convert(NormalGamma) -> Normal
""" ->
function convert(d::NormalGamma)
	μ = d.sums ./ d.n
	σ = sqrt((d.ssums / (d.n - 1)) - (μ^2))

	σ += 1e-8

	if isnan(σ) | isinf(σ)
		σ = 1.0
	end

	return Normal(μ, σ)
end

@doc doc"""
Convert ConjugatePostDistribution to Distribution.

convert(BinomialBeta) -> Binomial
""" ->
function convert(d::BinomialBeta)

	# smoothing
	if d.counts[1] == 0
		counts = d.counts[1] + 0.0000001
	elseif d.counts[1] == d.n
		counts = d.counts[1] - 0.0000001
	else
		counts = d.counts[1]
	end

	p = counts / d.n
	return Binomial(d.D, p)
end

function fit(dType::Type{GaussianWishart}, X::AbstractArray; useCov = true)

	(D, N) = size(X)

	μ0 = vec( mean(X, 2) )
	κ0 = 1.0
	ν0 = convert(Float64, D)
	Ψ = cov(X, vardim = 2) * 0.1

	if !useCov
		Ψ = eye(D) * 10
	end

	return GaussianWishart(μ0, κ0, ν0, Ψ)
end

function fit(dType::Type{NormalGamma}, X::AbstractArray)

	(D, N) = size(X)
	@assert D == 1

	return NormalGamma(μ = mean(X), λ = 10.0, α = 1.0, β = 1.0)
end

function fit(dType::Type{MultinomialDirichlet}, X::SparseMatrixCSC)

	(D, N) = size(X)

	c = vec(sum(X, 2))

	return MultinomialDirichlet(N, c, D, 1.0)
end

function fit(dType::Type{BinomialBeta}, X::AbstractArray)

	(D, N) = size(X)
	return BinomialBeta(D, α = 1.0, β = 1.0)
end

###
# add samples to distribution
###

function add_data(d::ConjugatePostDistribution, X)
   add_data!(deepcopy(d), X)
end

function add_data!(d::GaussianWishart, X)

    # process samples
    if ndims(X) == 1
        N = 1
        (D,) = size(X)
    else
        (D, N) = size(X)
    end

    if D != d.D
        throw(ArgumentError("Dimensions of X and distribution are not equal!"))
    end

    d.n += N
    d.sums += vec(sum(X, 2))
    d.ssums += X * X'

    d
end

function add_data!(d::NormalGamma, X)

	# process samples
	if ndims(X) == 1
		 N = 1
		 (D,) = size(X)
	else
		 (D, N) = size(X)
	end

	if D != 1
		 throw(ArgumentError("Data is not univariate!"))
	end

	d.n += N
	d.sums += sum(X)
	d.ssums += sum(X.^2)

	d

end

function add_data!(d::GaussianDiagonal, X)
	for (dim, dist) in enumerate(d.dists)
		add_data!(dist, X[dim,:])
	end

	d
end

function add_data!(d::NormalNormal, X::Real)

	d.n += 1
	d.sums += X
	d.ssums += X^2

	d

end

function add_data!(d::NormalNormal, X)

	# process samples
	if ndims(X) == 1
		 N = 1
		 (D,) = size(X)
	else
		 (D, N) = size(X)
	end

	if D != 1
		 throw(ArgumentError("Data is not univariate!"))
	end

	d.n += N
	d.sums += sum(X)
	d.ssums += sum(X.^2)

	d

end

function add_data!(d::MultinomialDirichlet, X)

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

    d
end

function add_data!(d::MultinomialDirichlet, X::SparseMatrixCSC{Int, Int})

    # process samples
    D = size(X, 1)
    N = size(X, 2)

    if D != d.D
        throw(ArgumentError("Dimensions of X don't match with Dimensions of Multinomial Distribution (D = $(d.D))!"))
    end

    d.n += N
    d.counts += sum(X, 2)
    d.dirty = true

    d
end

function add_data!(d::BinomialBeta, X)

	(D, N) = size(X)

	if D != d.D
		 throw(ArgumentError("Dimensions doesn't match! D (", D ,") != Beta-Binomial D (", d.D ,")"))
	end

	d.n += N
   d.counts += sum(X, 2)

	d
end

function add_data!(d::BernoulliBeta, X::AbstractArray)

	for x in X
		add_data!(d, x)
	end

	d
end

function add_data!(d::BernoulliBeta, X::Bool)

	if X
		d.successes += 1
	end

	d.n += 1

	d
end

###
# remove samples from distribution
###

function remove_data!(d::GaussianWishart, X)

    # process samples
    if ndims(X) == 1
        N = 1
        (D,) = size(X)
    else
        (D, N) = size(X)
    end

    if D != d.D
        throw(ArgumentError("Dimensions of X and distribution are not equal!"))
    end

    d.n -= N
    d.sums -= vec(sum(X, 2))
    d.ssums -= X * X'

    d
end

function remove_data!(d::NormalGamma, X)

    # process samples
    if ndims(X) == 1
        N = 1
        (D,) = size(X)
    else
        (D, N) = size(X)
    end

    if D != 1
        throw(ArgumentError("Data is not univariate!"))
    end

    d.n -= N
    d.sums -= sum(X)
    d.ssums -= sum(X.^2)

    d
end

function remove_data!(d::GaussianDiagonal, X)
	for (dim, dist) in enumerate(d.dists)
		remove_data!(dist, X[dim,:])
	end

	d
end

function remove_data!(d::NormalNormal, X::Real)

    d.n -= 1
    d.sums -= X
    d.ssums -= X^2

    d
end

function remove_data!(d::NormalNormal, X)

    # process samples
    if ndims(X) == 1
        N = 1
        (D,) = size(X)
    else
        (D, N) = size(X)
    end

    if D != 1
        throw(ArgumentError("Data is not univariate!"))
    end

    d.n -= N
    d.sums -= sum(X)
    d.ssums -= sum(X.^2)

    d
end

function remove_data!(d::MultinomialDirichlet, X)

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

    d
end

function remove_data!(d::MultinomialDirichlet, X::SparseMatrixCSC{Int, Int})

   # process samples
   D = size(X, 1)
   N = size(X, 2)

   if D != d.D
      throw(ArgumentError("Dimensions of X don't match with Dimensions of Multinomial Distribution (D = $(d.D))!"))
   end

   d.n -= N
   d.counts -= sum(X, 2)
   d.dirty = true

	 d

end

function remove_data!(d::BinomialBeta, X)

	(D, N) = size(X)

	if D != d.D
		 throw(ArgumentError("Dimensions doesn't match! D (", D ,") != Beta-Binomial D (", d.D ,")"))
	end

	d.n -= N
   d.counts -= sum(X, 2)

	d
end

function remove_data!(d::BernoulliBeta, X::AbstractArray)

	for x in X
		remove_data!(d, x)
	end

	d
end

function remove_data!(d::BernoulliBeta, X::Bool)

	if !isdistempty(d)
		if X
			d.successes -= 1
		end

		d.n -= 1
	end

	d
end

function remove_data(d::ConjugatePostDistribution, X)
    remove_data!(deepcopy(d), X)
end

###
# Check if distribution is empty (contains no samples)
###
function isdistempty(d::ConjugatePostDistribution)
    return d.n <= 0
end

###
# compute posterior predictive (unnormalized)
###

"Log PDF for GaussianWishart."
function logpred(d::GaussianWishart, x)

	if d.n == 0
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

		#p = v - shdfhdim * log1p(xx / df)
    p = 1/df * xx

    p = v - log((1 + p).^((df+1) / 2))

    return p

end

"Log PDF for NormalGamma."
function logpred(d::NormalGamma, x)

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
function logpred(d::MultinomialDirichlet, x)

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
function logpred(d::MultinomialDirichlet, x::SparseMatrixCSC{Int, Int})

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

"Log PMF for BinomialBeta."
function logpred(d::BinomialBeta, x)

   (D, N) = size(x)

   n = N + d.n
   k = d.counts + sum(x, 2)

   l1 = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
   l2 = lbeta(d.α + k, d.β + n - k) - lbeta(d.α, d.β)

   return l1 + l2

end

"Log PMF for BernoulliBeta."
function logpred(d::BernoulliBeta, X::AbstractArray)
	return Float64[logpred(d, x) for x in X]
end

function logpred(d::BernoulliBeta, X::Bool)

  	# posterior
		α = d.α + d.successes
		β = d.β + d.n - d.successes

   return log(α) - log(α + β)
end
