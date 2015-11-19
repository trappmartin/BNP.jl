include("vcm.jl")

# sample new sigma_g from InverseGamma with IG(a, b)
# see: supplemental of http://papers.nips.cc/paper/4579-a-nonparametric-variable-clustering-model
function random_sigma_g(B::VCMBuffer, γ_a::Float64, γ_b::Float64)

    Gm = (B.G.*B.C)

    a = γ_a + B.D  / 2.0
    b = γ_b + .5 * sum(Gm.*Gm)

    dist = InverseGamma(a, b)

    return sqrt( rand(dist) )

end

# sample new sigma_noise from InverseGamma with IG(a, b)
# see: supplemental of http://papers.nips.cc/paper/4579-a-nonparametric-variable-clustering-model
function random_sigma_noise(B::VCMBuffer, γ_a::Float64, γ_b::Float64)

    E = B.Y - (B.G.*B.C) * B.X

    a = ( γ_a + (B.D*B.N) ) / 2.0
    b = γ_b + .5 * trace(E'*E)

    dist = InverseGamma(a, b)

    return sqrt( rand(dist) )

end

# sample latent factor matrix X
# see: supplemental of http://papers.nips.cc/paper/4579-a-nonparametric-variable-clustering-model
function sample_latent_factors!(Y::Array, B::VCMBuffer)

    Gm = B.G .* B.C
    tem = Gm' / B.σ_noise^2

    sigma_x_p = try
            B.σ_x^(-2)
        catch
            1
        end

    λ = (B.σ_noise^(-2)) * sum( Gm .* Gm, 1 ) + sigma_x_p

    for n in 1:B.N

        μ = (tem * Y[:,n]) ./ λ'
        B.X[:,n] = λ'.^(-0.5) .* randn(B.K, 1) + μ

    end

    B
end

function sample_vcm_data(N::Int, D::Int, G::Array{Float64}, X::Array{Float64}, C::Array{Float64},
    mu_noise::Float64, sigma_noise::Float64)

    ϵ = mu_noise + sigma_noise * randn(D, N)

    return (G .* C) * X + ϵ

end

# sample cluster assignments c and lattent factor loadings G by integrating out g and instantiating X.
# See: Neal (2000) Algorithm 8 for details and
# supplemental of http://papers.nips.cc/paper/4579-a-nonparametric-variable-clustering-model
function gibbs_aux_assignment!(B::VCMBuffer; out_dim = -1)

    (D, N) = size(B.Y)

    if out_dim != -1
        perm = collect(out_dim)
    else
        perm = randperm(D)
    end

    auxDist = Normal(B.μ_x, B.σ_x)

    X2Cache = sum(B.X.^2, 2)

    for d in perm

        #println("out dim", out_dim)

        if out_dim == -1
            # get cluster of current item
            cluster = B.cc[d]

            #println("got cluster")

            # check if cluster is empty
            if sum(B.cc .== cluster) == 1

                #println("found empty cluster")

                B.K -= 1
                idx = find(B.cc .> cluster)
                B.cc[idx] -= 1

                #C = C[:,[1:cluster-1, cluster+1:end]]
                B.G = B.G[:,[1:cluster-1; cluster+1:end]]

                r = B.X[cluster,:]
                B.X = B.X[[1:cluster-1; cluster+1:end],:]

                #println("removed empty cluster")

            end
        else
            B.G = cat(1, B.G, zeros(1, B.K))
            B.cc = cat(1, B.cc, 0)
            #C = cat(1, C, zeros(Int, 1, K))
        end

        #println("generate cache...")

        Xm = reduce(hcat, map(m -> rand(auxDist, B.N), 1:B.m_aux))'
        X2m = sum(Xm.^2, 2)

        XX = cat(1, B.X, Xm)
        XX2Cache = cat(1, X2Cache, X2m)

        #println("prepare p..")

        # compute P(c_i = c | c_-i, y_i, ϕ_1, ..., ϕ_h ) according to Neal (2000)
        p = ones(B.K+B.m_aux) .* -Inf
        p[1:B.K] = log( map( c -> sum(B.cc .== c) / (B.α + D - 1), 1:B.K ) )
        p[B.K + 1:B.K + B.m_aux] = log( B.α / B.m_aux ) - log( B.α + D - 1 )

        Gsamps = zeros(1, B.K + B.m_aux)

        xy = XX * B.Y[d,:]'

        for k in 1:B.K + B.m_aux

            λ = XX2Cache[k] / B.σ_noise^2 + 1.0 / B.σ_g^2
            μ = ( xy[k] / B.σ_noise^2 + B.μ_g/B.σ_g^2 ) / λ

            p[k] = p[k] - .5 * ( log(λ) - λ * μ^2 )

            Gsamps[k] = μ + λ^(-.5) * randn()
        end

        #println("finished comp. of p...")

        # common normalization
        p = exp( p - maximum(p) )

        kk = rand_indices(p)

        #println("got new k")

        g = Gsamps[kk]

        #println("found g")

        if kk > B.K
            # open new cluster
            B.X = cat(1, B.X, XX[kk,:])
            #C = cat(2, C, zeros(Int, D, 1))
            B.G = cat(2, B.G, zeros(D, 1))

            B.K += 1
            kk = B.K

            X2Cache = sum(B.X.^2, 2)

        end

        if out_dim == -1
            B.G[d, kk] = g
            B.cc[d] = kk
            #C[d,:] = zeros(1, K)
            #C[d,kk] = 1
        else
            B.G[out_dim, kk] = g
            B.cc[out_dim] = kk
            #C[out_dim,:] = zeros(1, K)
            #C[out_dim,kk] = 1
        end

        #println("loop end")

    end

    B.C = spzeros(D, B.K)
    B.C[B.cc] = 1

    B
end
