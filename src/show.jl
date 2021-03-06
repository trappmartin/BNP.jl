# This code is from Distribution.jl!

distrname(d::ConjugatePostDistribution) = string(typeof(d))

show(io::IO, d::ConjugatePostDistribution) = show(io, d, fieldnames(typeof(d)))

# For some distributions, the fields may contain internal details,
# which we don't want to show, this function allows one to
# specify which fields to show.
#
function show(io::IO, d::ConjugatePostDistribution, pnames)
    uml, namevals = _use_multline_show(d, pnames)
    uml ? show_multline(io, d, namevals) : show_oneline(io, d, namevals)
end

function _use_multline_show(d::ConjugatePostDistribution, pnames)
    # decide whether to use one-line or multi-line format
    #
    # Criteria: if total number of values is greater than 8, or
    # there are matrix-valued params, we use multi-line format
    #
    namevals = _NameVal[]
    multline = false
    tlen = 0
    for (i, p) in enumerate(pnames)
        pv = d.(p)
        if !(isa(pv, Number) || isa(pv, NTuple) || isa(pv, AbstractVector))
            multline = true
        else
            tlen += length(pv)
        end
        push!(namevals, (p, pv))
    end
    if tlen > 8
        multline = true
    end
    return (multline, namevals)
end

function _use_multline_show(d::ConjugatePostDistribution)
    _use_multline_show(d, fieldnames(typeof(d)))
end

function show_oneline(io::IO, d::ConjugatePostDistribution, namevals)
    print(io, distrname(d))
    np = length(namevals)
    print(io, '(')
    for (i, nv) in enumerate(namevals)
        (p, pv) = nv
        print(io, p)
        print(io, '=')
        show(io, pv)
        if i < np
            print(io, ", ")
        end
    end
    print(io, ')')
end

function show_multline(io::IO, d::ConjugatePostDistribution, namevals)
    print(io, distrname(d))
    println(io, "(")
    for (p, pv) in namevals
        print(io, p)
        print(io, ": ")
        println(io, pv)
    end
    println(io, ")")
end
