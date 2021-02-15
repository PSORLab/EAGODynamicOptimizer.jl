# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# EAGODynamicOptimizer.jl
# See https://github.com/PSORLab/EAGODynamicOptimizer.jl
#############################################################################
# src/semiinfinite/sip_extension.jl
# Defines the SIPDynamicExt and extends EAGO SIP subroutines.
#############################################################################

mutable struct SIPDynamicExt{T} <: EAGO.ExtensionType
    llp_ext::DynamicExt{T}
    bnd_ext::DynamicExt{T}
end
function SIPDynamicExt(integrator)
    llp_ext = DynamicExt(integrator)
    bnd_ext = DynamicExt(integrator)
    return SIPDynamicExt{eltype(llp_ext)}(llp_ext, bnd_ext)
end

function get_ext(m::SIPDynamicExt{T}, s::S) where {T, S <: Union{LowerLevel1,LowerLevel2,LowerLevel3}}
    m.llp
end
function get_ext(m::SIPDynamicExt{T}, s::S) where {T, S <: Union{LowerProblem,UpperProblem,ResProblem}}
    m.bnd_ext
end

function EAGO.build_model(t::SIPDynamicExt{T}, a::A, s::S, p::SIPProblem) where {T, A <: AbstractSIPAlgo, S <: AbstractSubproblemType}
    model, v = EAGODynamicModel(get_ext(t,s))
    for (k,v) in EAGO.get_sip_kwargs(s,p)
        MOI.set(model, MOI.RawParameter(String(k)), v)
    end
    vL, vU, nv = EAGO.get_bnds(s,p)
    @variable(model, vL[i] <= v[i=1:nv] <= vU[i])
    return model, v
end

function EAGO.sip_llp!(t::SIPDynamicExt{T}, alg::A, s::S, result::SIPResult,
                       sr::SIPSubResult, prob::SIPProblem, cb::SIPCallback,
                       i::Int64, tol::Float64 = -Inf) where {T, A <: AbstractSIPAlgo, S <: AbstractSubproblemType}
    m, p = build_model(t, alg, s, prob)
    EAGO.set_tolerance!(t, alg, s, m, sr, i)

    # define the objective
    xbar = EAGO.get_xbar(t, alg, s, sr)
    add_supported_objective!(m, (y, p) -> -cb.gSIP[i](y, xbar, p))

    # optimize model and check status
    JuMP.optimize!(m)
    tstatus = JuMP.termination_status(m)
    rstatus = JuMP.primal_status(m)
    feas = llp_check(prob.local_solver, tstatus, rstatus)

    # fill buffer with subproblem result info
    psol = JuMP.value.(p)
    EAGO.load!(s, sr, feas, -JuMP.objective_value(m), -JuMP.objective_bound(m), psol)
    result.solution_time += MOI.get(m, MOI.SolveTime())

    return nothing
end

function EAGO.sip_bnd!(t::SIPDynamicExt{T}, alg::A, s::S, sr::SIPSubResult, result::SIPResult,
                       prob::SIPProblem, cb::SIPCallback) where {T, A <: AbstractSIPAlgo,
                                                                 S <: AbstractSubproblemType}

    # create JuMP model
    m, x = build_model(t, alg, s, prob)

    for i = 1:prob.nSIP
        ϵ_g = EAGO.get_eps(s, sr, i)
        disc_set = EAGO.get_disc_set(t, alg, s, sr, i)
        for j = 1:length(disc_set)
            pbar = disc_set[j]
            add_supported_constraint!(m, (y, x) -> cb.gSIP[i](y, x, pbar) + ϵ_g)
        end
    end

    # define the objective
    add_supported_objective!(m, (y, x) -> cb.f(y, x))

    # optimize model and check status
    JuMP.optimize!(m)
    t_status = JuMP.termination_status(m)
    r_status = JuMP.primal_status(m)
    feas = EAGO.bnd_check(prob.local_solver, t_status, r_status)

    # fill buffer with subproblem result info
    EAGO.load!(s, sr, feas, JuMP.objective_value(m), JuMP.objective_bound(m), JuMP.value.(x))
    result.solution_time += MOI.get(m, MOI.SolveTime())

    return nothing
end
