using Distributions

function random_concentration_parameter(alpha::Float64, gamma_a::Float64, gamma_b::Float64, N::Int, k::Int; maxiter = 1)

    # implementation according to escobar - west page 585
    # equations: 13 and 14.

    w = zeros(2)
    w[2] = N

    f0 = gamma_a + k
    f1 = gamma_a + k - 1

    for i in 1:maxiter

        eta = rand(Beta(alpha + 1, N))
        f3 = gamma_b - log(eta)

        w[1] = f1 / f3

        z = rand_indices(w)

        if z == 1
            alpha = rand(Gamma(f0, 1/f3))
        else
            alpha = rand(Gamma(f1, 1/f3))
        end
    end

    return alpha
end

function random_concentration_parameter(alpha::Float64, gamma_a::Float64, gamma_b::Float64, N::Array{Int}, k::Array{Int}; maxiter = 1)

    # implementation according to escobar - west page
    # using multiple groups (see

    totalK = sum(k)
    num = length(N)

    etas = zeros(num)

    for i in 1:maxiter

        for j in 1:num
            etas[j] = rand(Beta(alpha + 1, N[j]))
        end

        z = rand(num) .* (alpha + N) .< N

        g_a = gamma_a + totalK - sum(z)
        g_b = gamma_b - sum(log(etas))

        alpha = rand(Gamma(g_a, 1/g_b))
    end

    alpha
end

function rand_indices(prob::Array{Float64}; dim = 1)

    if ndims(prob) == 1
        N = 1
        (M,) = size(prob)
    else
        (N,M) = size(prob)
    end

    #if N == 1

        max = sum(prob)
        csum = 0.0
        thres = rand()

        for i in 1:length(prob)

            csum += prob[i]

            if csum >= thres * max
                return convert(Int, i)
            end
        end

#     else

#         # TODO: rewrite this part to be faster!

#         # prob needs to be normalized
#         cdf = cumsum(prob, dim)

#         # make sure cdf goes to 1
#         cdf[end] = 1.0

#         # rand produces unexpected dim array, has to be transposed\
#         return 1 + (sum((rand(N) .> cdf)', dim))
#     end
end

"Construct a polynomial from its root."
function poly{T}(r::Vector{T})
    n = length(r)
    c = zeros(T, n+1)
    c[1] = 1
    for j = 1:n
        c[2:j+1] = c[2:j+1]-r[j]*c[1:j]
    end
    return c
end

"Compute the signed Stirling number of first kind."
function stirlings1(n::Integer, k::Integer)
    p = poly(collect(0:(n-1)))
    p[n - k + 1]
end
