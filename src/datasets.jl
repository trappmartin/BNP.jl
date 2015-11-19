function generate_bloobs(n_features = 2, n_centers = 3, n_samples = 100)
    centers = @parallel (hcat) for i = 1:n_features
        rand(Uniform(-10, 10), n_centers)
    end

    cluster_std = @parallel (hcat) for i = 1:n_features
      rand(Uniform(0.5, 4), n_centers)
    end

    n_samples_per_center = ones(Integer, n_centers) * int(n_samples / n_centers)

    for i = 1:(n_centers % n_samples)
        n_samples_per_center[i] += 1
    end

    X = @parallel (hcat) for i = 1:n_centers
      rand(MvNormal(ones(n_features) .* centers[i], eye(n_features) .* cluster_std[i]), n_samples_per_center[i])
    end

    X = X';

    Y = @parallel (vcat) for i = 1:n_centers
      y = ones(n_samples_per_center[i]) * i
    end

    ids = [1:size(X)[1]]
    shuffle!(ids)

    X = X[ids,:]
    Y = Y[ids];

    return (X, Y)
end

function generateBarsDataset(img_size::Int, noise_level::Float64, num_per_mixture::Array{Float64}, num_group::Int, num_data::Int)

  numdim   = img_size^2;
  numbars  = img_size*2

  mix_theta = zeros(numdim, numbars)

  # add horizontal bars
  for i in 1:img_size
    img = ones(img_size, img_size) .* noise_level / img_size^2;
    img[:,i] = img[:,i] + 1/img_size;
    mix_theta[:,i] = img[:] / sum(img[:]);
  end

  # add vertical bars
  for i = 1:img_size
    img = ones(img_size, img_size) .* noise_level / img_size^2;
    img[i,:] = img[i,:] + 1/img_size;
    mix_theta[:,img_size+i] = img[:]/sum(img[:]);
  end

  # generate samples
  # Note: strange behavior of cumsum!!!
  cumbar = cumsum(num_per_mixture, 2)

  samples = Vector{Vector{Int}}(num_group)

  for g in 1:num_group

      # random number of mixture components
      n_bars = 1 + sum(rand() .> cumbar)
      k = randperm(numbars)[1:n_bars]

      # get weigths
      theta = mean(mix_theta[:,k], 2)

      # get samples (e.g. words)
      samples[g] = vec( 1 + sum(repmat(rand(num_data)', numdim, 1) .> repmat(cumsum(theta, 1), 1, num_data), 1) )
  end

  # return samples and bars
  return (samples, mix_theta)

end
