println("# Test Point Estimation using VI")

println(" * test data..")

Z = hcat(vec([1 1 2 2]),
	vec([1 1 2 2]),
	vec([1 2 2 2]),
	vec([2 2 1 1]))

println(" * compute psm")

psm = compute_psm(Z)

println(" * test variation_of_information_lb..")

cls = vec([1 1 1 1])

@test_approx_eq_eps BNP.variation_of_information_lb(cls, psm)[1] 0.9207175 1e-6

println(" * find point estimate")

Z = hcat(vec([1 1 2 2]),
	vec([1 1 2 2]),
	vec([1 2 2 2]),
	vec([1 1 2 2]))

psm = compute_psm(Z)

(idx, value) = point_estimate(psm, maxk = 2)

@test length(unique(idx)) == 2

println("# FINISHED")
