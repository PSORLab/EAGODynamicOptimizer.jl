"""
$(TYPEDEF)
"""
struct StateCalc{Z}
    integrator
    indx
end
function getindex(d::StateCalc{Interval{Float64}}, i::Int)
    lo = get(d.integrator, Bound{LOWER}(d.indx, TimeIndex(i)))
    hi = get(d.integrator, Bound{UPPER}(d.indx, TimeIndex(i)))
    return Interval{Float64}(lo, hi)
end
function getindex(d::StateCalc{MC{N,T}}, i::Int) where {N,T}
    lo = get(d.integrator, Bound{LOWER}(d.indx, TimeIndex(i)))
    hi = get(d.integrator, Bound{UPPER}(d.indx, TimeIndex(i)))
    cv = get(d.integrator, Relaxation{LOWER}(d.indx, TimeIndex(i)))
    cc = get(d.integrator, Relaxation{UPPER}(d.indx, TimeIndex(i)))
    cv_grad = get(d.integrator, Subgradient{LOWER}(d.indx, TimeIndex(i)))
    cc_grad = get(d.integrator, Subgradient{UPPER}(d.indx, TimeIndex(i)))
    return MC{N,T}(cv, cc, Interval{Float64}(lo, hi), cv_grad, cc_grad, false)
end

function getindex(d::Vector{StateCalc}, i::Int)
    return Interval{Float64}(lo, hi)
end

function (d::StateCalc{Interval{Float64}})(x::Float64)
    lo = get(d.integrator, Bound{LOWER}(d.indx, x))
    hi = get(d.integrator, Bound{UPPER}(d.indx, x))
    return Interval{Float64}(lo, hi)
end
function (d::StateCalc{MC{N,NS}})(x::Float64) where {N,NS}
    lo = get(d.integrator, Bound{LOWER}(d.indx, x))
    hi = get(d.integrator, Bound{UPPER}(d.indx, x))
    cv = get(d.integrator, Relaxation{LOWER}(d.indx, x))
    cc = get(d.integrator, Relaxation{UPPER}(d.indx, x))
    cv_grad = get(d.integrator, Subgradient{LOWER}(d.indx, x))
    cc_grad = get(d.integrator, Subgradient{UPPER}(d.indx, x))
    return MC{N,NS}(cv, cc, Interval{Float64}(lo, hi), cv_grad, cc_grad, false)
end
function (d::StateCalc{MC{N,Diff}})(x::Float64) where {N,NS}
    lo = get(d.integrator, Bound{LOWER}(d.indx, x))
    hi = get(d.integrator, Bound{UPPER}(d.indx, x))
    cv = get(d.integrator, Relaxation{LOWER}(d.indx, x))
    cc = get(d.integrator, Relaxation{UPPER}(d.indx, x))
    cv_grad = get(d.integrator, Gradient{LOWER}(d.indx, x))
    cc_grad = get(d.integrator, Gradient{UPPER}(d.indx, x))
    return MC{N,NS}(cv, cc, Interval{Float64}(lo, hi), cv_grad, cc_grad, false)
end

"""
$(TYPEDEF)
"""
struct SupportedScalarFunction <: MOI.AbstractSet
    g
    t::Parameter
    integrator
    nx::Int
end

"""
$(TYPEDEF)
"""
struct SupportedVectorFunction <: MOI.AbstractSet
    g!
    t::Parameter
    integrator
    nx::Int
    ng::Int
end

function bound(s::SupportedScalarFunction, lo::Vector{Float64}, hi::Vector{Float64})::Interval{Float64} where NP
    p = Interval.(lo, up)
    xc = StateCalc{Interval{Float64}}.(lo, hi)
    fintv = s.f(xc, pc, s.t)
    return fintv
end
function bound(s::SupportedVectorFunction, lo::Vector{Float64}, hi::Vector{Float64})::Vector{Interval{Float64}}
    p = Interval.(lo, up)
    xc = StateCalc{Interval{Float64}}.(lo, hi)
    fintv = s.f(xc, pc, s.t)
    return fintv
end

function seed_mc(val::SVector{N,Float64}, lo::SVector{N,Float64}, up::SVector{N,Float64},
                 tag::T) where {T <: EAGO.RelaxTag, N}
    p = Interval.(lo, up)
    np = length(p)
    mc = zeros(MC{N,T}, np)
    for i in 1:np
        mc[i] = MC{N,T}.(val[i], p[i], i)
    end
    return mc
end

for sType in (SupportedScalarFunction, SupportedVectorFunction)
    @eval function set_state_calc(s::sType, nx::Int, val::Val{N}, tag::T) where {T <: EAGO.RelaxTag, N}
        xc = zeros(StateCalc{MC{N,T}}, nx)
        for i in 1:nx
            xc[i] = StateCalc{MC{N,T}}.(s.integrator,i)
        end
        return xc
    end
end

function relax(s::SupportedScalarFunction, val::SVector{N,Float64}, lo::SVector{N,Float64},
               up::SVector{N,Float64}, tag::T)::SVector{MC{N,T}} where {T<:EAGO.RelaxTag, N}
    pc = seed_mc(val, lo, up, tag)
    xc = set_state_calc(s, s.nx, Val{N}(), tag)
    gmc = s.g(xc, pc, s.t)
    return fmc
end

function relax(s::SupportedVectorFunction, val::SVector{N,Float64}, lo::SVector{N,Float64},
               up::Vector{Float64}, tag::T)::Vector{MC{N,T}} where {T<:EAGO.RelaxTag, N}
    pc = seed_mc(val, lo, up, tag)
    xc = set_state_calc(s, s.nx, Val{N}(), tag)
    gmc = zeros(MC{N,T}, s.ng)
    s.g!(gmc, xc, pc, s.t)
    return gmc
end
