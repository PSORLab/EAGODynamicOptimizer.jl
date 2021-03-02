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
mutable struct SIPDynamicExt{P,T} <: EAGO.ExtensionType
    prob::P
    temp_prob::P
    integrator_factory
    llp_ext::DynamicExt{T}
    bnd_ext::DynamicExt{T}
end
function SIPDynamicExt(integrator_factory, prob)
    llp_ext = DynamicExt(integrator_factory(prob))
    bnd_ext = DynamicExt(integrator_factory(prob))
    return SIPDynamicExt{eltype(prob),eltype(llp_ext)}(prob, prob,
                                                       integrator_factory,
                                                       llp_ext, bnd_ext)
end


function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPResRev, s::LowerLevel1, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.eps_l[i])
end
function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPResRev, s::LowerLevel2, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.eps_u[i])
end
function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPResRev, s::LowerProblem, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.lbd.tol)
end
function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPResRev, s::UpperProblem, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.ubd.tol)
end

function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPHybrid, s::LowerLevel1, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.eps_l[i])
end
function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPHybrid, s::LowerLevel2, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.eps_u[i])
end
function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPHybrid, s::LowerProblem, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.lbd.tol)
end
function EAGO.set_tolerance!(t::SIPDynamicExt, alg::SIPHybrid, s::UpperProblem, m::JuMP.Model, sr::SIPSubResult, i::Int)
    EAGO.set_tolerance_inner!(EAGO.DefaultExt(), alg, s, m, sr.ubd.tol)
end

function EAGO.get_xbar(t::SIPDynamicExt, alg::A, s::P, sr::EAGO.SIPSubResult) where {A <: EAGO.AbstractSIPAlgo, P <: Union{LowerLevel1,LowerLevel2,LowerLevel3}}
    EAGO.get_xbar(EAGO.DefaultExt(), alg, s, sr)
end

function get_ext(m::SIPDynamicExt{T}, s::S) where {T, S <: Union{LowerLevel1,LowerLevel2,LowerLevel3}}
    m.llp_ext
end
function get_ext(m::SIPDynamicExt{T}, s::S) where {T, S <: Union{LowerProblem,UpperProblem,ResProblem}}
    m.bnd_ext
end

function EAGO.build_model(t::SIPDynamicExt{T}, a::A, s::S, p::SIPProblem) where {T, A <: AbstractSIPAlgo, S <: AbstractSubproblemType}
    vL, vU, nv = EAGO.get_bnds(s,p)
    ext = get_ext(t,s)
    DBB.setall!(ext.integrator, DBB.ParameterBound{Lower}(), vL)
    DBB.setall!(ext.integrator, DBB.ParameterBound{Upper}(), vU)
    model, v = EAGODynamicModel(ext)
    for (k,v) in EAGO.get_sip_kwargs(s,p)
        MOI.set(model, MOI.RawParameter(String(k)), v)
    end
    return model, v
end

function EAGO.sip_llp!(t::SIPDynamicExt{T}, alg::A, s::S, result::SIPResult,
                       sr::SIPSubResult, prob::SIPProblem, cb::SIPCallback,
                       i::Int64, tol::Float64 = -Inf) where {T, A <: AbstractSIPAlgo, S <: AbstractSubproblemType}
    m, p = build_model(t, alg, s, prob)
    EAGO.set_tolerance!(t, alg, s, m, sr, i)
    xbar = EAGO.get_xbar(t, alg, s, sr)
    @show xbar

    # update rhs and x0 function in problem
    llp_ext = get_ext(t,s)

    # define the objective
    ode_prob = t.prob
    fnew = (dy,y,p,t) -> ode_prob.f.f(dy,y,xbar,p,t)
    x0new = p -> ode_prob.x0(p,xbar)
    temp_prob = ODERelaxProb(fnew, ode_prob.tspan, x0new, ode_prob.pL, ode_prob.pU)
    DBB.set!(temp_prob, DBB.get(ode_prob, DBB.SupportSet()))
    obj_integrator = t.integrator_factory(temp_prob)
    add_supported_objective!(m, xbar,
                             (y, p) -> -cb.gSIP[i](y, xbar, p),
                             obj_integrator)
    # optimize model and check status
    JuMP.optimize!(m)
    tstatus = JuMP.termination_status(m)
    rstatus = JuMP.primal_status(m)
    feas = EAGO.llp_check(prob.local_solver, tstatus, rstatus)

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
            ode_prob = t.prob
            x0new = u -> ode_prob.x0(u,pbar)
            fnew = (dy,y,u,t) -> ode_prob.f.f(dy,y,u,pbar,t)
            temp_prob = ODERelaxProb(fnew, ode_prob.tspan, x0new, ode_prob.pL, ode_prob.pU)
            DBB.set!(temp_prob, DBB.get(ode_prob, DBB.SupportSet()))
            cons_integrator = t.integrator_factory(temp_prob)
            add_supported_constraint!(m,
                                      (y, x) -> cb.gSIP[i](y, x, pbar) + ϵ_g,
                                      cons_integrator)
        end
    end

    # define the objective
    add_supported_objective!(m, Float64[], (y,x) -> cb.f(x), NoIntegrator())

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
