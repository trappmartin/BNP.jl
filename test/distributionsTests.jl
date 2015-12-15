println("# Test Distributions")

# Beta-Binomial
println(" * Beta-Binomial")

d = BinomialBeta(1) # one dimensional

println(" * - add data")

add_data!(d, [true true true false false])
@test d.n == 5
@test d.D == 1
@test sum(d.counts) == 3

println(" * - remove data")

remove_data!(d, [true true true])
@test d.n == 2
@test d.D == 1
@test sum(d.counts) == 0

println(" * - log pdf")

d = BinomialBeta(1, α = 3, β = 4)
@test_approx_eq_eps logpred(d, [true true true false false])[1] -1.53 1e-2


println("# FINISHED")
