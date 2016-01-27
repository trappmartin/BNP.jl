
@doc doc"""
point_estimate_avg(psm::Array{Float64, 2})

	Find optimal partition which minimizes the lower bound to the Variation of Information
	obtain from Jensen's inequality where the expectation and log are reversed.

	Code based on R implementation by Sara Wade <sara.wade@eng.cam.ac.uk>

""" ->
function point_estimate_hclust(psm::Array{Float64, 2}; maxk = -1, method = :average)

	havg = hclust(1-psm, method)
	cls = reduce(hcat, map(k -> cutree(havg; k = k), 1:maxk))

	# compute Variation of Information lower bound
	vi = variation_of_information_lb(cls, psm)

	idx = findfirst(minimum(vi) .== vi)

	if ndims(cls) == 2
		cl = cls[:,idx]
	else
		cl = cls
	end

	return (cl, minimum(vi))

end

@doc doc"""
point_estimate(psm::Array{Float64, 2})

	Find optimal partition which minimizes the lower bound to the Variation of Information
	obtain from Jensen's inequality where the expectation and log are reversed.

	Code based on R implementation by Sara Wade <sara.wade@eng.cam.ac.uk>

""" ->
function point_estimate(psm::Array{Float64, 2}; method = :avg, maxk = -1)

	methods = [:avg, :comp, :greedy]

	@assert method in methods "Method must be :avg , :comp or :greedy."

	if maxk == -1
		maxk = convert(Int, ceil(size(psm)[1] / 4.0))
	end

	if method == :avg
		return point_estimate_hclust(psm, maxk = maxk, method = :average)
	elseif method == :comp
		return point_estimate_hclust(psm, maxk = maxk, method = :complete)
	else
		println("not implemented")
	end
end

@doc doc"""
variation_of_information()
	Computes the lower bound to the posterior expected Variation of Information
""" ->
function variation_of_information_lb(cls::Array{Int, 2}, psm)

	N = size(psm)[1]
	F = zeros(size(cls)[2])

	for ci in 1:size(cls)[2]
		c = cls[:,ci]
		for i in 1:N
			ind = c .== c[i]
			F[ci] += (log2(sum(ind)) +log2(sum(psm[i,:])) -2 * log2(sum(ind' .* psm[i,:]))) / N
		end
	end

	return F

end

@doc doc"""
variation_of_information()
	Computes the lower bound to the posterior expected Variation of Information
""" ->
function variation_of_information_lb(cls::Vector{Int}, psm)

	N = size(psm)[1]
	F = zeros(1)

	for i in 1:N
		ind = cls .== cls[i]
		F[1] += (log2(sum(ind))+log2(sum(psm[i,:]))-2*log2(sum(ind' .* psm[i,:]))) / N
	end

	return F

end
