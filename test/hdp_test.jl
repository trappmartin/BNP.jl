println("Test randnumtable()")
weights = ones(10, 10)
table = ones(Int, 10, 10)

@test weights == BNP.randnumtable(weights, table)
