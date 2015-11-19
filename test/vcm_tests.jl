println("# VCM TESTS...")

# run demo code at first
println(" * try demo...")


D = 40
K = 2
N = 100

C = zeros(D, K)
C[1:20, 1] = 1
C[21:40, 2] = 1

cc = zeros(1,D)
cc[1:20] = 1
cc[21:40] = 2

mu_x = 0.0;
sigma_x = 1.0;
mu_g = 0.0;
sigma_g = 1.0;
mu_noise = 0.0;
sigma_noise = 0.1;

# set G and X (factor loadings, and latent factors) of test data
G = mu_g + sigma_g * randn(D, K)
X = mu_x + sigma_x * randn(K, N)

# set noise
ϵ = mu_noise + sigma_noise * randn(D,N);

# test data
Y = (G .* C) * X + ϵ;

train(VCM(), Gibbs(), IncrementalInitialisation(), Y);
@time train(VCM(), Gibbs(), IncrementalInitialisation(), Y);



println("# FINISHED")
