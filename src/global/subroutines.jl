# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# EAGODynamicOptimizer.jl
# See https://github.com/PSORLab/EAGODynamicOptimizer.jl
#############################################################################
# src/subroutines.jl
# Defines the DynamicExt and extends EAGO subroutines used in branch and bound.
#############################################################################

function supports_affine_relaxation(integrator)
    supports(integrator, Relaxation{Lower}()) &&
    supports(integrator, Relaxation{Upper}()) &&
    supports(integrator, Subgradient{Lower}()) &&
    supports(integrator, Subgradient{Upper}())
end

function EAGO.presolve_global!(t::DynamicExt{T}, m::EAGO.Optimizer) where T

    EAGO.presolve_global!(EAGO.DefaultExt(), m)

    m._working_problem._variable_count = m._branch_variable_count
    copyto!(m._working_problem._variable_info, m._input_problem._variable_info)

    # add a map of branch/node index to variables in the continuous solution
    for i = 1:m._working_problem._variable_count
        if m._working_problem._variable_info[i].is_fixed
            m._branch_variables[i] = false
            continue
        end
        if m._branch_variables[i]
            push!(m._branch_to_sol_map, i)
        end
    end

    # creates reverse map
    m._sol_to_branch_map = zeros(m._working_problem._variable_count)
    for i = 1:length(m._branch_to_sol_map)
        j = m._branch_to_sol_map[i]
        m._sol_to_branch_map[j] = i
    end

    # add storage for objective cut
    m._working_problem._objective_saf.terms = fill(MOI.ScalarAffineTerm{Float64}(0.0,
                                                   MOI.VariableIndex(1)),
                                                   m._branch_variable_count)

    # set up for extension
    np = m.ext_type.np
    nx = m.ext_type.nx
    m.ext_type.p_intv = zeros(Interval{Float64}, np)
    last_obj = m.ext_type.obj

    integrator = m.ext_type.integrator
    support_set = get(integrator, DBB.SupportSet())
    nt = length(support_set.s)
    if supports_affine_relaxation(integrator)
        m.ext_type = DynamicExt(integrator, np, nx, nt, zero(MC{np, NS}))
        m.ext_type.obj = last_obj
        load_check_support!(t, m, support_set, nt, nx,  zero(MC{np, NS}))
        for i = 1:nt
            push!(m.ext_type.lower_storage.x_set_traj.v, zeros(MC{np,NS}, nx))
        end
    else
        m.ext_type = DynamicExt(integrator, np, nx, nt, zero(Interval{Float64}))
        m.ext_type.obj = last_obj
        load_check_support!(t, m, support_set, nt, nx, zero(Interval{Float64}))
        for i = 1:nt
            push!(m.ext_type.lower_storage.x_set_traj.v, zeros(Interval{Float64}, nx))
        end
    end
    for cons in t.cons
        push!(m.ext_type.cons, SupportedFunction(cons, Float64[], cons.params))
    end
    for i = 1:nt
        push!(m.ext_type.x_traj.v, zeros(Float64, nx))
    end
    m.ext_type.lower_storage.x_set_traj.nt = nt
    m.ext_type.lower_storage.x_set_traj.nx = nx

    m._presolve_time = time() - m._parse_time

    return nothing
end

function lower_bound_problem!(::Val{false}, t::DynamicExt, opt::EAGO.Optimizer)

    # reset box used to evaluate relaxation
    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds
    integrator = t.integrator

    setall!(integrator, ParameterBound{Lower}(), lvbs)
    setall!(integrator, ParameterBound{Upper}(), uvbs)

    # set reference point to evaluate relaxation
    @__dot__ opt._current_xref = 0.5*(lvbs + uvbs)
    setall!(integrator, ParameterValue(), opt._current_xref)
    @__dot__ t.p_intv = Interval(lvbs, uvbs)
    @__dot__ t.lower_storage.p_set = t.p_intv

    # relaxes pODE
    relax!(integrator)

    for i = 1:t.nt
        support_time = t.obj.support[i]
        get(t.lo[i], integrator, Bound{Lower}(support_time))
        get(t.hi[i], integrator, Bound{Upper}(support_time))
    end
    load_intervals!(t.x_intv, t.lo, t.hi, t.nt)

    # loads trajectory
    for i = 1:nt
        t.lower_storage.x_set_traj.v[i] .= t.x_intv[i]
    end

    feasible = true
    for i = 1:length(t.cons)
        cons_interval = t.cons[i].f(t.lower_storage.x_set_traj, t.lower_storage.p_set)
        if t.cons[i].f(t.lower_storage.x_set_traj, t.lower_storage.p_set) > 0.0
            feasible = false
            break
        end
    end
    opt._lower_objective_value = lo(t.lower_storage.obj_set)
    opt._lower_solution = opt._current_xref
    opt._lower_feasibility = feasible
    return nothing
end

function lower_bound_problem!(::Val{true}, t::DynamicExt, opt::EAGO.Optimizer)

    # reset box used to evaluate relaxation
    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds
    integrator = t.integrator

    setall!(integrator, ParameterBound{Lower}(), lvbs)
    setall!(integrator, ParameterBound{Upper}(), uvbs)

    # set reference point to evaluate relaxation
    @__dot__ opt._current_xref = 0.5*(lvbs + uvbs)
    setall!(integrator, ParameterValue(), opt._current_xref)
    @__dot__ t.p_intv = Interval(lvbs, uvbs)
    @__dot__ t.lower_storage.p_set = MC{t.np,NS}(opt._current_xref, t.p_intv, 1:t.np)

    # relaxes pODE
    relax!(integrator)

    for i = 1:t.nt
        support_time = t.obj.support[i]
        get(t.lo[i], integrator, Bound{Lower}(support_time))
        get(t.hi[i], integrator, Bound{Upper}(support_time))
        get(t.cv[i], integrator, Relaxation{Lower}(support_time))
        get(t.cc[i], integrator, Relaxation{Upper}(support_time))
        get(t.cv_grad[i], integrator, Subgradient{Lower}(support_time))
        get(t.cc_grad[i], integrator, Subgradient{Upper}(support_time))
    end
    load_intervals!(t.x_intv, t.lo, t.hi, t.nt)

    # loads trajectory
    load_trajectory!(t.lower_storage.x_set_traj, t.cv,
                     t.cc, t.x_intv, t.cv_grad, t.cc_grad)

    # computes objective
    t.lower_storage.obj_set = t.obj(t.lower_storage.x_set_traj, t.lower_storage.p_set)

    saf_temp = opt._working_problem._objective_saf
    saf_temp.constant = t.lower_storage.obj_set.cv
    for i = 1:t.np
        coeff = @inbounds t.lower_storage.obj_set.cv_grad[i]
        saf_temp.terms[i] = MOI.ScalarAffineTerm{Float64}(coeff, MOI.VariableIndex(i))
        pv = opt._current_xref[i]
        saf_temp.constant = saf_temp.constant - coeff*pv
    end

    relaxed_optimizer = opt.relaxed_optimizer
    MOI.set(relaxed_optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf_temp)
    MOI.set(relaxed_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    for i = 1:length(t.cons)
        cons_mc = t.cons[i].f(t.lower_storage.x_set_traj, t.lower_storage.p_set)
        sat_vec = MOI.ScalarAffineTerm{Float64}[MOI.ScalarAffineTerm{Float64}(0.0, MOI.VariableIndex(i)) for i = 1:t.np]
        saf_temp_cons = MOI.ScalarAffineFunction{Float64}(sat_vec, 0.0)
        for j = 1:t.np
            coeff = @inbounds cons_mc.cv_grad[j]
            saf_temp_cons.terms[j] = MOI.ScalarAffineTerm{Float64}(coeff, MOI.VariableIndex(j))
            pv = opt._current_xref[j]
            saf_temp_cons.constant = saf_temp.constant - coeff*pv
        end
        constant_value = saf_temp_cons.constant
        saf_temp_cons.constant = 0.0
        MOI.add_constraint(relaxed_optimizer, saf_temp_cons, MOI.LessThan{Float64}(constant_value))
    end

    MOI.optimize!(relaxed_optimizer)

    opt._lower_termination_status = MOI.get(relaxed_optimizer, MOI.TerminationStatus())
    opt._lower_result_status = MOI.get(relaxed_optimizer, MOI.PrimalStatus())
    valid_flag, feasible_flag = EAGO.is_globally_optimal(opt._lower_termination_status, opt._lower_result_status)

    if valid_flag
        opt._cut_add_flag = true
        opt._lower_feasibility = true
        aff_obj = MOI.get(relaxed_optimizer, MOI.ObjectiveValue())
        if aff_obj > lo(t.lower_storage.obj_set)
            opt._lower_objective_value = MOI.get(relaxed_optimizer, MOI.ObjectiveValue())
            for i = 1:opt._working_problem._variable_count
                opt._lower_solution[i] = MOI.get(relaxed_optimizer, MOI.VariablePrimal(), opt._relaxed_variable_index[i])
            end
        else
            opt._lower_objective_value = lo(t.lower_storage.obj_set)
            opt._lower_solution = opt._current_xref
            opt._lower_feasibility = true
        end
    else
        lower_interval_problem!(t, opt)
    end
    return nothing
end

function EAGO.lower_problem!(t::DynamicExt, opt::EAGO.Optimizer)
    lower_bound_problem!(Val(supports_affine_relaxation(t.integrator)), t, opt)
end

function set_dual_trajectory!(::Val{NP}, t::DynamicExt) where NP
    DBB.getall!(t.gradient_temp, t.integrator, DBB.Gradient{Nominal}())
    temp = zeros(np)
    out = zeros(Partials{Float64,NP},nx)
    for i = 1:nt
        for k = 1:nx
            for j = 1:np
                temp[j] = t.gradient_temp[j][i,k]
            end
            out[k] = Partials{Float64,NP}(temp)
        end
        t.upper_storage.x_set_traj.v[i] .= out
    end
    return nothing
end

function evaluate_dynamics(t, param, p)
    DBB.setall!(t.integrator, DBB.ConstantParameterValue(), param)
    new_point = t.p_val != p
    if new_point
        integrate!(t.integrator)
        for i = 1:t.nt
            support_time = t.obj.support[i]
            get(t.x_val[i], t.integrator, DBB.Value(support_time))
            t.x_traj.v[i] .= t.x_val[i]
        end
        seeds = construct_seeds(Partials{NP,Float64})
        @__dot__ t.upper_storage.p_set = Dual{TAG,Float64,NP}(p, seeds)
        t.p_val .= p
    end
    return new_point
end

function obj_wrap(t::DynamicExt, p)
    new_eval = evaluate_dynamics(t, t.obj.params, p)
    t.obj_val = t.obj.f(t.x_traj, t.p_val)
    return t.obj_val
end

function cons_wrap(t::DynamicExt, params, i, p)
    new_eval = evaluate_dynamics(t, t.cons[i].params, p)
    t.cons_val[i] = t.cons[i].f(t.x_traj, t.p_val)
    return t.cons_val[i]
end

function ∇obj_wrap!(::Val{NP}, t::DynamicExt, out, p) where NP
    new_eval = evaluate_dynamics(t, t.obj.params, p)
    set_dual_trajectory!{NP}(Val{NP}(),t)
    obj_dual = t.obj.f(t.upper_storage.x_set_traj, t.upper_storage.p_set)
    out .= partials(obj_dual)
    return nothing
end

function ∇cons_wrap!(::Val{NP}, t::DynamicExt, param, out, i, p) where NP
    new_eval = evaluate_dynamics(t, t.cons[i].params, p)
    set_dual_trajectory!(Val{NP}(),t)
    cons_dual = t.cons[i].f(t.upper_storage.x_set_traj, t.upper_storage.p_set)
    out .= partials(cons_dual)
    return nothing
end

function upper_problem_obj_only!(q::DynamicExt, opt::EAGO.Optimizer)
    t = opt.ext_type

    # get all at particular points???
    DBB.set!(t.integrator, DBB.LocalSensitivityOn(), false)

    integrate!(t.integrator)
    getall!(t.p_val, t.integrator, DBB.ParameterValue())
    for i = 1:t.nt
        support_time = t.obj.support[i]
        get(t.x_val[i], t.integrator, DBB.Value(support_time))
        t.x_traj.v[i] .= t.x_val[i]
    end
    t.obj_val = t.obj.f(t.x_traj, t.p_val)
    opt._upper_objective_value = t.obj_val
    opt._upper_feasibility = true
    @__dot__ opt._upper_solution = t.p_val

    return nothing
end

function EAGO.upper_problem!(q::DynamicExt, opt::EAGO.Optimizer)
    isempty(q.cons) && return upper_problem_obj_only!(q, opt)

    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds
    np = q.np

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    @variable(model, lvbs[i] <= p[i=1:q.np] <= uvbs[i])

    DBB.set!(q.integrator, DBB.LocalSensitivityOn(), true)

    # define the objective
    JuMP.register(model, :obj, np, (p...) -> obj_wrap(q, p...),
                                   (out, p...) -> ∇obj_wrap!(Val{np}(),q,out,p...),
                                   (out, p...) -> error("Hessian called but is currently disabled."))
    if isone(q.np)
        nl_obj = :(obj($(p[1])))
    else
        nl_obj = Expr(:call)
        push!(nl_obj.args, :obj)
        for i in 1:q.np
            push!(nl_obj.args, p[i])
        end
    end
    set_NL_objective(model, MOI.MIN_SENSE, nl_obj)

    for (cons, i) in enumerate(q.cons)
        cons_udf_sym = Symbol("cons_udf_$i")
        JuMP.register(model, cons_udf_sym, np,
                             (p...) -> cons_wrap(q, i, p...),
                             (out, p...) -> ∇cons_wrap!(Val{np}(),q,i,out,p...),
                             (out, p...) -> error("Hessian called but is currently disabled."))

        gic = Expr(:call)
        push!(gic.args, cons_udf_sym)
        for i in 1:q.np
            push!(gic.args, p[i])
        end
        JuMP.add_NL_constraint(model, :($gic <= 0))
    end

    JuMP.optimize!(model)
    t_status = JuMP.termination_status(model)
    r_status = JuMP.primal_status(model)
    feas = EAGO.is_feasible(t_status, r_status)

    opt._upper_objective_value = feas ? objective_value(model) : Inf
    opt._upper_feasibility = feas
    @__dot__ opt._upper_solution = value(p)
    return nothing
end

function EAGO.preprocess!(t::DynamicExt, p::Optimizer)
    p._preprocess_feasibility = true
    return nothing
end

function EAGO.postprocess!(t::DynamicExt, p::Optimizer)
    p._postprocess_feasibility = true
    return nothing
end

EAGO.cut_condition(t::DynamicExt, p::Optimizer) = false
