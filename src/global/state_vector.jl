# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# EAGODynamicOptimizer.jl
# See https://github.com/PSORLab/EAGODynamicOptimizer.jl
#############################################################################
# src/state_vector.jl
# Utilities for writing functions which reference support time and index
# when using Base.getindex.
#############################################################################

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
    d.v[tindx][indx]
end

function setindex!(d::Trajectory{T}, val::T, i::Int64, j::Int64) where T<:Number
    d.v[i][j] = val
    return nothing
end

function extract_static_val(::Type{Val{N}}, grad::Vector{Matrix{Float64}}, x::Int64, t::Int64) where N
    return SVector{N,Float64}(ntuple(p -> grad[t][x,p], Val{N}()))
end

function extract_static_vector(::Type{Val{N}}, ::Type{Val{NX}}, grad::Vector{Matrix{Float64}}, i::Int64) where {N,NX}
    return ntuple(x -> extract_static_val(Val{N}, grad, x, i), Val{NX}())
end

function load_trajectory!(d::Trajectory{MC{N,T}}, cv::Vector{Vector{Float64}},
                          cc::Vector{Vector{Float64}}, intv::Vector{Vector{Interval{Float64}}},
                          cv_grad::Vector{Matrix{Float64}}, cc_grad::Vector{Matrix{Float64}}) where {N, T<:RelaxTag}
    for i = 1:d.nt
        cvg = extract_static_vector(Val{N}, Val{d.nx}, cv_grad, i)
        ccg = extract_static_vector(Val{N}, Val{d.nx}, cc_grad, i)
        for j = 1:d.nx
            d.v[i][j] = MC{N,T}.(cv[i][j], cc[i][j], intv[i][j], cvg, ccg, false)
        end
    end
    return nothing
end
