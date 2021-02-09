mutable struct SIPDynamicExt <: EAGO.ExtensionType
    llp_ext::DynamicExt
    bnd_ext::DynamicExt
end
function get_ext(m::SIPDynamicExt, s::T) where T <: Union{LowerLevel1,LowerLevel2,LowerLevel3}
    get_ext.llp
end
function get_ext(m::SIPDynamicExt, s::T) where T <: Union{LowerProblem,UpperProblem,ResProblem}
    get_ext.bnd_ext
end

function build_model(t::SIPDynamicExt, a::A, s::S, p::SIPProblem) where {A <: AbstractSIPAlgo, S <: AbstractSubproblemType}
    model, v = EAGODynamicModel(get_ext(t,s), EAGO.Optimizer)
    for (k,v) in get_sip_kwargs(s,p)
        MOI.set(model, MOI.RawParameter(String(k)), v)
    end
    @variable(model, vL[i] <= v[i=1:nv] <= vU[i])
    return model, v
end

function sip_llp!(t::SIPDynamicExt, alg::A, s::S, result::SIPResult,
                  sr::SIPSubResult, prob::SIPProblem, cb::SIPCallback,
                  i::Int64, tol::Float64 = -Inf)
    m, p = build_model(t, alg, s, prob)
    set_tolerance!(t, alg, s, m, sr, i)

    # define the objective
    xbar = get_xbar(t, alg, s, sr)
    add_supported_objective!(m, (y, p) -> -cb.gSIP[i](y, xbar, p))

    # optimize model and check status
    JuMP.optimize!(m)
    tstatus = JuMP.termination_status(m)
    rstatus = JuMP.primal_status(m)
    feas = llp_check(prob.local_solver, tstatus, rstatus)

    # fill buffer with subproblem result info
    psol = JuMP.value.(p)
    load!(s, sr, feas, -JuMP.objective_value(m), -JuMP.objective_bound(m), psol)
    result.solution_time += MOI.get(m, MOI.SolveTime())

    return nothing
end

function sip_bnd!(t::SIPDynamicExt, alg::A, s::S, sr::SIPSubResult, result::SIPResult,
                  prob::SIPProblem, cb::SIPCallback) where {A <: AbstractSIPAlgo,
                                                            S <: AbstractSubproblemType}

    # create JuMP model
    m, x = build_model(t, alg, s, prob)

    for i = 1:prob.nSIP
        ϵ_g = get_eps(s, sr, i)
        disc_set = get_disc_set(t, alg, s, sr, i)
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
    feas = bnd_check(prob.local_solver, t_status, r_status)

    # fill buffer with subproblem result info
    load!(s, sr, feas, JuMP.objective_value(m), JuMP.objective_bound(m), JuMP.value.(x))
    result.solution_time += MOI.get(m, MOI.SolveTime())

    return nothing
end
