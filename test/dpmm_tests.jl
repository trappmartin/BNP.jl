println("# DPM TESTS...")

# run demo code at first
println(" * try demo...")

using DataFrames

# number of samples per cluster
N = 200

# create clusters
Data = DataFrame( x = randn(N), y = randn(N), class = "cluster1" )

# append second cluster
append!(Data, DataFrame( x = randn(N) + 5, y = randn(N) + 5, class = "cluster2" ));

D = 2 # 2 dimensional data
N = 400 # number of data points

# data matrix
X = zeros(N, D)

X[:,1] = convert(Array, Data[:x])
X[:,2] = convert(Array, Data[:y])

# init base distribution parameters
mu0 = mean(X, 1)
kappa0 = 9.0
nu0 = 5.0
Sigma0 = eye(D) * 10

# base distribution and concentration parameter (Gaussian with Normal Inverse Wishart Prior)
H = WishartGaussian(vec(mu0), kappa0, nu0, Sigma0)

# train Dirichlet Process Mixture Model
models = train(DPM(H), Gibbs(), RandomInitialisation(k = 10), X);
train(DPM(H), Gibbs(maxiter = 1000), KMeansInitialisation(), X)

# use $\alpha = 10.0$
train(DPM(H, α = 10.0), Gibbs(), KMeansInitialisation(), X)

# use Random Initialisation
train(DPM(H), Gibbs(), RandomInitialisation(), X)

# use 50 iterations and 10 burnin runs
train(DPM(H), Gibbs(maxiter = 50, burnin = 10),
	KMeansInitialisation(), X)

# access the log likelihood values of all iteration
LLH = map(model -> model.energy, models)

# access the weights of all iterations
W = map(model -> model.weights, models)

# access the distributions of all iterations
D = map(model -> model.distributions, models)


println("# FINISHED")
