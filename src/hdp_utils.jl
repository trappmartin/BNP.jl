function randnumtable(weights::Array{Float64}, table::Array{Int})

    numtable = zeros(Int, size(table))

    B = unique( table )
    w = log( weights )

    for i in 1:length(B)

        max = B[i]
        if max > 0

            m = 1:max

            stirnums = map( x -> abs(stirlings1(max, x)), m)
            stirnums /= maximum(stirnums)

            for (idx, j) in enumerate(find(table .== max))

                clike = m .* w[idx]
                clike = cumsum(stirnums .* exp(clike - maximum(clike)))

                numtable[j] = 1+sum(rand() * clike[max] .> clike)
            end


        end

    end

    return numtable

end
