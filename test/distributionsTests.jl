println("# Test Distributions")

# Multinomial-Dirichlet
println(" * Dirichlet - Multinomial")

d = DirichletMultinomial(5, 1.0)

println(" * - add data")

add_data!(d, [2, 3, 3])

@test d.n == 3
@test d.counts[2] == 1
@test d.counts[3] == 2

println(" * - add sparse data")

add_data!(d, sparse([1,0,1,0,0]))

@test d.n == 4
@test d.counts[1] == 1
@test d.counts[3] == 3

println(" * - remove data")

remove_data!(d, [3])

@test d.n == 3
@test d.counts[3] == 2

println(" * - remove sparse data")

remove_data!(d, sparse([1,0,0,0,0]))

@test d.n == 2
@test d.counts[1] == 0

println(" * - log pdf")
@test logpred(d, [3])[1] == logpred(d, sparse([0,0,1,0,0]))[1]

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

# Normal-Gamma
println(" * Gamma-Normal")

d = GammaNormal()

(" * - add data")

x = 1.0

add_data!(d, x)
@test d.n == 1

add_data!(d, vec([1.0, 2.0, 1.5, 1.0]))
@test d.n == 5

println(" * - remove data")

remove_data!(d, 2.0)
@test d.n == 4

println(" * - log pdf")

x2 = [2.0]
@test logpred(d, x)[1] > logpred(d, x2)[1]

# Normal-Gamma
println(" * GaussianDiagonal")

d = GaussianDiagonal{NormalNormal}(NormalNormal[NormalNormal() for i in 1:2])

println(" * - add data")
add_data!(d, rand(2))
@test d.n == 1

println("# FINISHED")
