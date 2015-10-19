"Dirichlet Process Mixture Model Hyperparameters"
immutable DPMHyperparam

  γ_a::Float64
  γ_b::Float64

  "default values"
  DPMHyperparam() = new(1.0, 1.0)

end

"Dirichlet Process Mixture Model Data Object"
immutable DPMData

  # Energy
  energy::Float64

  # Dirichlet concentration parameter
  α::Float64

  # Distributions
  G::Array{ConjugatePostDistribution}

  # Assignments
  Z::Array{Int}

end

type DPMBuffer

  # ids used for random access
  ids::Array{Int}

  # dimensionality of data
  D::Int

  # number of samples
  N::Int

  # samples
  X::Array{Float64}

  # assignments
  Z::Array{Int}

  # number of samples per cluster
  C::Array{Int}

  # number of active cluster
  K::Int

  # distributions
  G::Array{ConjugatePostDistribution}

  # base distribution
  G0::ConjugatePostDistribution

  # concentration parameter
  alpha::Float64
end

"DPM initialization using K-Means."
function init_kmeans_dpmm(X::Array{Float64}, G0::ConjugatePostDistribution; k = 2, maxiterations = 10)

  (D, N) = size(X)

  R = Clustering.kmeans(X, k; maxiter = maxiterations)
  Z = assignments(R)

  G = Array{ConjugatePostDistribution}(N)

  for c in 1:N
    idx = find(Z .== c)

    if length(idx) > 0
      G[c] = add_data(G0, X[:,idx])
    else
      G[c] = deepcopy(G0)
    end
  end

  return (Z, G)
end

"DPM initialization using random assignments."
function init_random_dpmm(X::Array{Float64}, G0::ConjugatePostDistribution; k = 2)

  (D, N) = size(X)

  Z = 1 + (randperm(N) % k)

  G = Array{ConjugatePostDistribution}(N)

  for c in 1:N
    idx = find(Z .== c)

    if length(idx) > 0
      G[c] = add_data(G0, X[:,idx])
    else
      G[c] = deepcopy(G0)
    end
  end

  return (Z, G)
end

"Single iteration of collabsed Gibbs sampling using CRP."
function cgibbs_crp_dpmm!(B::DPMBuffer)

  # randomize data
  shuffle!(B.ids)

  z = -1
  k = -1

  for index in B.ids

    x = view(B.X, : ,index)

    # get assignment
    z = B.Z[index]

    # remove sample from cluster
    remove_data!(B.G[z], x)

    # remove cluster assignment
    B.C[z] -= 1

    # udpate number of active clusters if necessary
    if B.C[z] < 1
      B.K -= 1
    end

    # remove cluster assignment
    z = -1

    # compute posterior predictive
    # compute priors using chinese restaurant process
    # see: Samuel J. Gershman and David M. Blei, A tutorial on Bayesian nonparametric models.
    # In Journal of Mathematical Psychology (2012)
    p = ones(B.K + 1) * -Inf
    k2id = ones(Int, B.K + 1)

    j = 1
    for i in 1:length(B.C)
      if B.C[i] >= 1

        llh = logpred( B.G[i], x )
        crp = log( B.C[i] / (B.N + B.alpha - 1) )

        p[j] = llh + crp
        k2id[j] = i

        j += 1

      end
    end

    k2id[B.K + 1] = k2id[B.K] + 1

    p[B.K + 1] = logpred(B.G0, x) + log( B.alpha / (B.N + B.alpha - 1) )
    p = exp(p - maximum(p))

    k = k2id[rand_indices(p)]

    if k > length(B.G)
      # add new cluster
      Gk = add_data(B.G0, x)
      B.G = cat(1, B.G, Gk)
      B.C = cat(1, B.C, 0)
    else
      # add to cluster
      add_data!(B.G[k], x)
    end

    if k > B.K
      B.K += 1
    end

    B.C[k] += 1
    B.Z[index] = k
  end

  B
end

"Training Dirichlet Process Mixture Model using collabsed Gibbs sampling"
function train_cgibbs_dpmm(X::Array, G0::ConjugatePostDistribution, Z::Array{Int}, G::Array{ConjugatePostDistribution}, hyper::DPMHyperparam; alpha = 1.0, burnin = 0, thinout = 1, maxiter = 100)

  (D, N) = size(X)

  results = DPMData[]

  # construct C
  zz = unique(Z)

  C = zeros(Int, N)
  for i = 1:N
    C[i] = sum(Z .== i)
  end

  K = length(zz)

  # construct Buffer
  B = DPMBuffer(
    collect(1:N), # ids
    D,
    N,
    X,
    Z,
    C,
    K,
    G,
    G0,
    alpha)

  # burn-in phase
  for iter in 1:burnin
    # run one gibbs iteration using chinese restaurant process
    B = cgibbs_crp_dpmm!(B)

    # sample concentration parameter from G(1, 1)
    B.alpha = random_concentration_parameter(B.alpha, hyper.γ_a, hyper.γ_b, B.N, B.K)
  end

  # Gibbs sweeps

  for iter in 1:maxiter

    for t in 1:thinout
      # run one gibbs iteration using chinese restaurant process
      B = cgibbs_crp_dpmm!(B)

      # sample concentration parameter from G(1, 1)
      B.alpha = random_concentration_parameter(B.alpha, hyper.γ_a, hyper.γ_b, B.N, B.K)
    end

    # TODO: compute energy
    e = 0.0

    # record results
    push!(results, DPMData(e, deepcopy(B.alpha), deepcopy(B.G), deepcopy(B.Z)) )
  end

  return results

end
