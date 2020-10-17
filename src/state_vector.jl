mutable struct Trajectory{T<:Number}
    v::Vector{Vector{T}}
    time_dict::Dict{Float64,Int64}
end

function set_support!(d::Trajectory, support::DBB.SupportSet)
    empty!(d.time_dict)
    for (indx, val) in enumerate(support.s)
        d.time_dict[val] = indx
    end
    return nothing
end

function getindex(d::Trajectory{T}, indx::Int64, t::Float64) where T<:Number
    tindx = d.time_dict[t]
    v[tindx][indx]
end

#function setindex(d::Trajectory{T}, val::T, i::Int64, j::Int64)
    #d.v[] =
#end
