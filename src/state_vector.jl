mutable struct Trajectory{T<:Number}
    v::Vector{Vector{T}}
    time_dict::Dict{Float64,Int64}
    nt::Int64
    nx::Int64
end
function Trajectory{T}(nt::Int64, nx::Int64) where T<:Number
    v = Vector{T}[]
    for i = 1:nt
        push!(v, zeros(T, nx))
    end
    Trajectory{T}(v, Dict{Float64,Int64}(), nt, nx)
end
Trajectory{T}() where T<:Number = Trajectory{T}(0, 0)

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

function setindex!(d::Trajectory{T}, val::T, i::Int64, j::Int64) where T<:Number
    d.v[i][j] = val
    return nothing
end

function extract_static(::Type{Val{N}}, grad::Vector{Matrix{Float64}}, i::Int64, j::Int64) where N
    return SVector{N,Float64}(ntuple(k -> grad[k][j,i], Val(N)))
end

function load_trajectory!(d::Trajectory{MC{N,T}}, cv::Vector{Vector{Float64}},
                          cc::Vector{Vector{Float64}}, l::Vector{Vector{Float64}},
                          u::Vector{Vector{Float64}}, cv_grad::Vector{Matrix{Float64}},
                          cc_grad::Vector{Matrix{Float64}}) where {N,T<:RelaxTag}
    for i = 1:d.nt
        for j = 1:d.nx
            d.cvg[i][j] = extract_static(Val{N}, cc_grad, i, j)
            d.ccg[i][j] = extract_static(Val{N}, cc_grad, i, j)
        end
        @__dot__ d.v[i] = MC{N,T}(cv[i], cc[i], Interval(xL[i], xU[i]), cvg[i], ccg[i], false)
    end
    return nothing
end

function load_trajectory!(d::Trajectory{Interval{Float64}},
                          xL::Vector{Vector{Float64}},
                          xU::Vector{Vector{Float64}})
    for i = 1:d.nt
        for j = 1:d.nx
            d.v[i][j] = Interval(xL[j,i], xU[j,i])
        end
    end
    return nothing
end

function load_trajectory!(d::Trajectory{Float64}, x::Vector{Vector{Float64}})
    for i = 1:d.nt
        for j = 1:d.nx
            d.v[i][j] = x[j,i]
        end
    end
    return nothing
end
