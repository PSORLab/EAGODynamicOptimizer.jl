function build_model(t::DynamicExt, a::A, s::S, p::SIPProblem) where {A <: AbstractSIPAlgo,
                                                                      S <: AbstractSubproblemType}
    model = Model(EAGO.Optimizer)
    for (k,v) in get_sip_kwargs(a,s,p)
        MOI.set(model, MOI.RawParameter(String(k)), v)
    end
    vL, vU, nv = get_bnds(s,a,p)
    @variable(model, vL[i] <= v[i=1:nv] <= vU[i])
    return model, v
end

function sip_llp!(t::DynamicExt, alg::A, s::S, result::SIPResult,
                  sr::SIPSubResult, prob::SIPProblem, cb::SIPCallback,
                  i::Int64, tol::Float64 = -Inf) where {A <: AbstractSIPAlgo,
                                                        S <: AbstractSubproblemType}

    # build the model
    m, p = build_model(t, alg, s, prob)
    EAGO.set_tolerance!(t, alg, s, m, sr, i)

    # define the objective
    g(p...) = cb.gSIP[i](xbar, p)
    add_supported_objective!(m, g)

    # optimize model and check status
    JuMP.optimize!(m)
    tstatus = JuMP.termination_status(m)
    rstatus = JuMP.primal_status(m)
    feas = EAGO.llp_check(prob.local_solver, tstatus, rstatus)

    # fill buffer with subproblem result info
    load!(s, buffer, feas, -JuMP.objective_bound(m), JuMP.value(x))
    result.solution_time += MOI.get(m, MOI.SolveTime())

    return nothing
end

function sip_bnd!(t::DynamicExt, alg::A, s::S, sr::SIPSubResult, result::SIPResult,
                  prob::SIPProblem, cb::SIPCallback) where {A <: AbstractSIPAlgo,
                                                            S <: AbstractSubproblemType}

    # create JuMP model
    m, x = build_model(t, alg, s, prob)
    disc_set = get_disc_set(s, prob)

    for i = 1:prob.nSIP
        ϵ_g = get_eps(s, sr, i)
        for j = 1:length(disc_set)
            g = (x...) -> cb.gSIP[i](x, disc_set[j][i]) + ϵ_g
            add_supported_constraint!(m, g)
        end
    end

    # define the objective
    obj = (x...) -> cb.f(x)
    add_supported_objective!(m, obj)

    # optimize model and check status
    JuMP.optimize!(m)
    t_status = JuMP.termination_status(m)
    r_status = JuMP.primal_status(m)
    feas = bnd_check(prob.local_solver, t_status, r_status, eps_g)

    # fill buffer with subproblem result info
    load!(s, buffer, feas, JuMP.objective_bound(m), JuMP.value(x))
    result.solution_time += MOI.get(m, MOI.SolveTime())

    return nothing
end
