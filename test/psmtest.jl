println("# Test Estimate of Posterior Similarity Matrix")

println(" * generate test data")

Z = hcat(vec([1 1 2 2]),
	vec([1 1 2 2]),
	vec([1 2 2 2]),
	vec([2 2 1 1]))

println(" * compute psm")

psm = compute_psm(Z)

truePSM = [1.00 0.75 0.00 0.00;
						0.75 1.00 0.25 0.25;
						0.00 0.25 1.00 1.00;
						0.00 0.25 1.00 1.00]

@test all(psm .== truePSM)

println(" * generate test data 2")

Z = hcat(vec([1 1 2 2]),
	vec([1 1 2 2]),
	vec([1 2 2 2]),
	vec([2 2 1 1]),
	vec([1 1 2 1]))

println(" * compute psm")

psm = compute_psm(Z)

truePSM = [1.0  0.8  0.0  0.2;
						0.8  1.0  0.2  0.4;
						0.0  0.2  1.0  0.8;
						0.2  0.4  0.8  1.0]

@test all(psm .== truePSM)

println("# FINISHED")
