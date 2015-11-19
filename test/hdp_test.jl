println("# HDP TESTS...")

# run demo code at first
println(" * try demo...")

# define size of bar image
img_size = 5

# define amount of noise
noise_level = 0.01

# define probabilities of generating a particular number of bars
num_per_mixture = [0 ones(1,3)] ./ 3

# define number of groups (J)
num_group = 40

# define number of data items drawn from each group
num_data = 50;

# generate data
(X, bars) = generateBarsDataset(img_size, noise_level, num_per_mixture, num_group, num_data);

# Dimensionality of the data
D = img_size * img_size

# We assume a Multinomial Distribution with a Dirichlet Prior as base distribution
H = MultinomialDirichlet(D, 1.0)

# train a Hierarchical Dirichlet Process Mixture Model
# guessing 10 shared Distributions
models = train(HDP(H), Gibbs(), RandomInitialisation(k = 10), X);

# access the log likelihood values of all iteration
LLH = map(model -> model.energy, models)

# access the weights of all iterations
W = map(model -> model.weights, models)

# access the distributions of all iterations
D = map(model -> model.distributions, models)

println(" * test randnumtable()")
weights = ones(10, 10)
table = ones(Int, 10, 10)

@test weights == BNP.randnumtable(weights, table)

println("# FINISHED")
