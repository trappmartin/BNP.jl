"Hierarchical Dirichlet Process Mixture Model Hyperparameters"
immutable HDPHyperparam <: AbstractHyperparam

  γ_a::Float64
  γ_b::Float64

  α_a::Float64
  α_b::Float64

  "default values"
  HDPHyperparam() = new(1.0, 1.0, 1.0, 1.0)

end

"Hierarchical Dirichlet Process Mixture Model Data Object"
type HDPData <: AbstractModelData

  # Energy
  energy::Float64

  # Distributions
  G::Array{ConjugatePostDistribution}

  # Assignments
  Z::Array{Array}

end

type HDPBuffer <: AbstractModelBuffer

  # samples
  X::Array{Array}

  # number of groups
  N0::Int

  # indecies of groups
  N0idx::Vector{Int}

  # number of samples per group
  Nj::Vector{Int}

  # indecies of samples per group
  Njidx::Array{Vector}

  # assignments
  Z::Array{Array}

  # number of active cluster
  K::Int

  # number of samples per group per cluster
  C::Array{Int, 2}

  # total number of tables
  totalnt::Array

  # number of clusters per table
  classnt::Array

  # distributions
  G::Array{ConjugatePostDistribution}

  # base distribution
  G0::ConjugatePostDistribution

  # concentration parameter
  α::Float64

  # concentration parameters of clusters
  β::Array{Float64}

  # gamma
  γ::Float64
end


"HDP initialization using random assignments."
function init_random_hdp(samples::Array{Array}, G0::ConjugatePostDistribution; k::Int = 10)

    # TODO make sure the code is type-stable and Julia 0.4 compatible

    Z = Array(Array, length(samples))
    G = ConjugatePostDistribution[]

    for i in 1:length(samples)

        cc = 1 + (collect(1:length(samples[i])) % k)
        Z[i] = cc[randperm(length(samples[i]))]

        for c in 1:k
            idx = find(Z[i] .== c)

            if length(idx) > 0
              if size(G, 1) < c
                  Gc = add_data(G0, vec(int(samples[i][idx])))
                  push!(G, Gc)
              else
                  G[c] = add_data(G[c], vec(int(samples[i][idx])))
              end
            end
        end

    end

    (Z, G)

end

"Single Gibbs sweep of HDP training using beta variables."
function gibbs_beta_hdp!(B::HDPBuffer)

    shuffle!(B.N0idx)

    prob = zeros(B.K + 1) * -Inf

    for j in B.N0idx

        # get samples of group
        data = B.X[j]
        z = B.Z[j]

        shuffle!(B.Njidx[j])

        for i in B.Njidx[j]

            cluster = z[i]

            # remove data item from model
            remove_data!(B.G[cluster], data[i])
            B.C[cluster, j] -= 1

            if isdistempty(B.G[cluster])

                # remove cluster
                B.G = B.G[[1:cluster-1, cluster+1:end]]

                for jj in 1:B.N0
                    B.Z[jj][B.Z[jj] .> cluster] -= 1
                end

                B.β = B.β[[1:cluster-1, cluster+1:end]]

                B.classnt = B.classnt[:,[1:cluster-1, cluster+1:end]]
                B.totalnt = B.totalnt[[1:cluster-1, cluster+1:end]]'

                B.K -= 1

                B.C = B.C[[1:cluster-1, cluster+1:end],:]
                prob = zeros(B.K + 1) * -Inf
            end

            # compute log likelihood
            for k in 1:B.K
              llh = logpred(B.G[k], data[i])[1]
              prior = B.C[k, j] + B.β[k] * B.α
              prob[k] = llh + log( prior )
            end

            #prob[1:B.K] = pmap(k -> logpred(B.G[k], data[i])[1] + log(B.C[k, j] + B.β[k] * B.α), 1:B.K)

            prob[B.K + 1] = logpred(B.G0, data[i])[1] + log( B.β[B.K+1] * B.α )

            prob = exp(prob - maximum(prob))

            c = rand_indices(prob)

            # add data to model
            B.Z[j][i] = c

            if c > B.K
                # create new cluster
                B.K += 1
                B.G = cat(1, B.G, deepcopy(B.G0))
                b = rand(Dirichlet([1, B.γ]))
                b = b * B.β[end]
                B.β = cat(1, B.β, 1)
                B.β[end-1:end] = b
                B.C = cat(1, B.C, zeros(Int, 1, B.N0))
                prob = zeros(B.K + 1) * -Inf
            end

            B.C[c, j] += 1
            add_data!(B.G[c], [data[i]])

        end

    end

    # sample number of tables
    kk = maximum([0, B.K - length(B.totalnt)])
    B.totalnt = cat(2, B.totalnt - sum(B.classnt, 1), zeros(Int, 1, kk))
    B.classnt = randnumtable(B.α .* B.β[:,ones(Int, B.N0)]', B.C')
    B.totalnt = B.totalnt + sum(B.classnt, 1)

    # update beta weights
    a = zeros(B.K + 1)
    a[1:end-1] = B.totalnt
    a[end] = B.γ

    B.β = rand(Dirichlet(a))

    B
end

function compute_energy!(B::HDPData, X::Array{Array})

  E = 1e-10
  numl = 0

  for j in length(X)

      # get samples of group
      data = X[j]

      for i in length(data)
        pp = 0.0
        c = 0

        for k = 1:length(B.G)
          p = exp( logpred( B.G[k], data[i] ) ) * B.W[j, k]

          # only sum over actual values (excluding nans)
          if p == p
            pp += p
            c += 1
          end
        end

        E += pp
        numl += 1

      end

  end

  B.energy = log( E / numl )

end

function train_gibbs_hdp(X::Array{Array}, G0::ConjugatePostDistribution, Z::Array{Array}, G::Array{ConjugatePostDistribution}, hyper::HDPHyperparam, K::Int;
                           α = 1.0, γ = 1.0, burnin = 0, thinout = 1, maxiter = 100)

  N0 = length(X)
  N0idx = collect(1:N0)

  Nj = [length(x) for x in X]
  Njidx = [collect(1:N) for N in Nj]

  C = reduce(hcat, [StatsBase.counts(z, 1:K) for z in Z])

  # init step
  β = ones(K) ./ K
  classnt = randnumtable(α * β[:,ones(Int, N0)]', C')
  totalnt = sum(classnt, 1)

  # set alpha vector of Dirichlet Distribution to sample β
  a = zeros(K + 1)
  a[1:end-1] = totalnt
  a[end] = γ

  # update beta
  β = rand(Dirichlet(a));

  B = HDPBuffer(
    X,
    N0,
    N0idx,
    Nj,
    Njidx,
    Z,
    K,
    C,
    totalnt,
    classnt,
    G,
    G0,
    α,
    β,
    γ)

  # burn-in phase
  for iter in 1:burnin
    gibbs_beta_hdp!(B)

    # TODO: this should be type stable not like this!!!
    totalnt = sum(B.classnt, 1)
    B.γ = random_concentration_parameter(B.γ, hyper.γ_a, hyper.γ_b, sum(B.totalnt), B.K, maxiter=10)
    B.α = random_concentration_parameter(B.α, hyper.α_a, hyper.α_b, B.Nj, B.totalnt)
  end

  results = HDPData[]

  #println("starting")
  #addprocs(CPU_CORES)

  for iter = 1:maxiter

    for t in 1:thinout
      gibbs_beta_hdp!(B)
      # TODO: this should be type stable not like this!!!
      totalnt = sum(B.classnt, 1)
      B.γ = random_concentration_parameter(B.γ, hyper.γ_a, hyper.γ_b, sum(B.totalnt), B.K, maxiter=10)
      B.α = random_concentration_parameter(B.α, hyper.α_a, hyper.α_b, B.Nj, B.totalnt)
    end


    # TODO: compute energy
    e = 0.0

    # record results
    push!(results, HDPData(e, deepcopy(B.G), deepcopy(B.Z)) )

    #println("iteration: ", iter, " clusters: ", B.K)

  end

  return results

end
