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

function extract_static(::Type{Val{N}}, grad::Vector{Matrix{Float64}}, i::Int64, j::Int64)
    return SVector{N,Float64}(ntuple(k -> grad[k][j,i], Val(N)))
end

function load_trajectory!(d::Trajectory{MC{N,T}}, cv::Matrix{Float64}, cc::Matrix{Float64},
                          l::Matrix{Float64}, u::Matrix{Float64}, cv_grad::Vector{Matrix{Float64}},
                          cc_grad::Vector{Matrix{Float64}}) where {N,T<:RelaxTag}
    for i = 1:d.nt
        for j = 1:d.nx
            cvg = extract_static(Val{N}, cc_grad, i, j)
            ccg = extract_static(Val{N}, cc_grad, i, j)
            d.v[i][j] = MC{N,T}(cv[j,i], cc[j,i], Interval(xL[j,i], xU[j,i])), cvg, ccg, false)
        end
    end
    return nothing
end

function load_trajectory!(d::Trajectory{Interval{Float64}}, xL::Matrix{Float64}, xU::Matrix{Float64})
    for i = 1:d.nt
        for j = 1:d.nx
            d.v[i][j] = Interval(xL[j,i], xU[j,i])
        end
    end
    return nothing
end

function load_trajectory!(d::Trajectory{Float64}, x::Matrix{Float64})
    for i = 1:d.nt
        for j = 1:d.nx
            d.v[i][j] = x[j,i]
        end
    end
    return nothing
end
