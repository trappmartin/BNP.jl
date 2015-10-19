using ArrayViews

###
# add samples to distribution
###

function add_data!(d::GaussianWishart, X)

    n = d.n
    sums = d.sums
    ssums = d.ssums

    # process samples
    if ndims(X) == 1
        N = 1
        (D,) = size(X)
    else
        (D, N) = size(X)
    end

    if D != d.D
        println("D: ", D, " != Dist D: ", d.D)
        throw(ArgumentError("Dimensions of X and distribution are not equal!"))
    end

    d.n += N
    d.sums += vec(sum(X, 2))
    d.ssums += X * X'

    d
end

function add_data(d::GaussianWishart, X)
   add_data!(deepcopy(d), X)
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

    d
end

function add_data(d::MultinomialDirichlet, X)
   add_data!(deepcopy(d), X)
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

function remove_data(d::GaussianWishart, X)
    remove_data!(deepcopy(d), X)
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

    d
end

function remove_data(d::MultinomialDirichlet, X)
    remove_data!(deepcopy(d), X)
end

###
# Check if distribution is empty (contains no samples)
###
function isdistempty(d::GaussianWishart)
    return d.n <= 0
end

function isdistempty(d::MultinomialDirichlet)
    return d.n <= 0
end

###
# compute posterior predictive (unnormalized)
###

function logpred(d::GaussianWishart, x)

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

    d = Distributions.MvTDist(nu - d.D + 1, mu, Sigma * ((kappa + 1) / (kappa * (nu - d.D + 1))))
    return(Distributions.logpdf(d, x))

end

function logpred(d::MultinomialDirichlet, x)

    N = length(x)

    # construct sparse vector
    xx = spzeros(d.D, N)
    for i in 1:N
      xx[x[i]] += 1
    end

    m = sum(xx)
    mi = nnz(xx)

    l1 = lgamma(m + 1) - sum(lgamma(mi + 1))
    l2 = lgamma(d.alpha0 + d.n) - lgamma(d.alpha0 + d.n + m)
    l3 = sum( lgamma( d.alpha0 / d.D + d.counts + xx ) - lgamma( d.alpha0/d.D + d.counts ) )

    return l1 + l2 + l3
end
