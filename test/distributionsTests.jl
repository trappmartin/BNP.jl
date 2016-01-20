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

# student-t Distribution
println(" * Generalized student-t Distribution")

df = 1
mean = 0
sigma = 1

d = Distributions.TDist(df)
@test_approx_eq_eps BNP.tlogpdf(0, df, mean, sigma) Distributions.logpdf(d, 0) 1e-4

df = 1
mean = 1
sigma = 1

d = Distributions.TDist(df)
@test_approx_eq_eps BNP.tlogpdf(0, df, mean, sigma) Distributions.logpdf(d, 0 - mean) 1e-4

df = 1
mean = 0
sigma = 10

d = Distributions.TDist(df)
#@test_approx_eq_eps BNP.tlogpdf(1, df, mean, sigma) Distributions.logpdf(d, 1 / sigma) - log(sigma) 1e-4


# Normal-Gamma
println(" * Normal-Gamma")

d = NormalGamma()

(" * - add data")

x = [1.0]

add_data!(d, x)
@test d.n == 1

add_data!(d, [1.0 2.0 1.5 1.0])
@test d.n == 5

println(" * - remove data")

remove_data!(d, [2.0])
@test d.n == 4

println(" * - log pdf")

x2 = [2.0]
@test logpred(d, x)[1] > logpred(d, x2)[1]


println("# FINISHED")
