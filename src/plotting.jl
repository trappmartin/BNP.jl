using Gadfly, Distributions

function plot2D(G::Array{ConjugatePostDistribution}; 
    clim::(Float64, Float64, Float64, Float64) = (-10.0, 10.0, -10.0, 10.0), res::Int=50)
    
    sx = (clim[2]-clim[1])/(res-1)
    sy = (clim[4]-clim[3])/(res-1)
    A = Array(Float64,2,res^2)
    for (i,a) in enumerate(clim[1]:sx:clim[2])
        for (j,b) in enumerate(clim[3]:sy:clim[4])
            A[:,(i-1)*res+j] = [a,b]
        end
    end

    p = zeros(res * res)
    for d in G
        
        μ = vec(d.sums / d.n)
        if μ != μ
            μ = vec(zeros(size(d.sums)))
        end
        
        Σ = (d.ssums / d.n) + (μ * μ')
        
        
        
        p += pdf(MvNormal(μ, Σ), A)
    end
    
    z= reshape(p,res,res)
    return Gadfly.plot(z=z,x=[clim[1]:sx:clim[2]],y=[clim[3]:sy:clim[4]],Gadfly.Geom.contour)
end