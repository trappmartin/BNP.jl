@doc doc"""
compute_ps(Z::Array{Int, 2})

Estimate of Posterior Similarity Matrix

Assuming Z to be [Trails x Observations]

""" ->
function compute_psm(Z::Array{Int, 2})

	(N, M) = size(Z)

	valuerange = collect(1:M)
	inRange = true

	for n in 1:N
		for m in 1:M
				if !(Z[n, m] in valuerange)
					inRange = false
				end
		end
	end

	@assert inRange "Values are not in range: All values must be larger equal 1 and smaller equal "

	psm = zeros(N, N)

	for m in 1:M
		for i in 1:N
			for j in 1:N
				if Z[i, m] == Z[j, m]
					psm[i, j] += 1
				end
			end
		end
	end

	return psm ./ M

end
