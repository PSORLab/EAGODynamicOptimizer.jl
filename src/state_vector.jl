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

function extract_static_val(::Type{Val{N}}, grad::Vector{Matrix{Float64}}, i::Int64, j::Int64) where N
    return SVector{N,Float64}(ntuple(k -> grad[k][i,j], Val(N)))
end

function extract_static_vector(::Type{Val{N}}, vnx::Val{NX}, grad::Vector{Matrix{Float64}}, i::Int64) where {N,NX}
    return ntuple(k -> extract_static_val(Val{N}(), grad, i, k), vnx)
end

function load_trajectory!(d::Trajectory{MC{N,T}}, cv::Vector{Vector{Float64}},
                          cc::Vector{Vector{Float64}}, l::Vector{Vector{Float64}},
                          u::Vector{Vector{Float64}}, cv_grad::Vector{Matrix{Float64}},
                          cc_grad::Vector{Matrix{Float64}}) where {N, T<:RelaxTag}
    for i = 1:d.nt
        cvg = extract_static_vector(Val{N}(), Val{d.nx}(), cv_grad, i)
        ccg = extract_static_vector(Val{N}(), Val{d.nx}(), cc_grad, i)
        @__dot__ d.v[i] = MC{N,T}(cv[i], cc[i], Interval(xL[i], xU[i]), d.cvg, ccg, false)
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
