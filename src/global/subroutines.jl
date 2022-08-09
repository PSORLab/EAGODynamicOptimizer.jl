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


struct NoIntegrator <: DBB.AbstractODERelaxIntegrator
end


function supports_affine(integrator)
    b = supports(integrator, Relaxation{Lower}())
    b &= supports(integrator, Relaxation{Upper}())
    b &= supports(integrator, Subgradient{Lower}())
    b &= supports(integrator, Subgradient{Upper}())
    return b
end

function EAGO.presolve_global!(t::DynamicExt{T}, m::GlobalOptimizer) where T


    println("ran global presolve...")
    EAGO.presolve_global!(DefaultExt(), m)

    # add storage for objective cut
    m._working_problem._objective_saf.terms = fill(MOI.ScalarAffineTerm{Float64}(0.0,
                                                   MOI.VariableIndex(1)),
                                                   m._branch_variable_count)

    # set up for extension
    np = m.ext.np
    nx = m.ext.nx
    m.ext.p_intv = zeros(Interval{Float64}, np)
    last_obj = m.ext.obj

    # loads lower problem based on extension integrator
    integrator = m.ext.integrator
    support_set = DBB.get(integrator, DBB.SupportSet())
    nt = length(support_set.s)
    @show nt
    ext_type = DynamicExt(integrator, np, nx, nt, zero(MC{np, NS})) #Makes a new DynamicExt? TODO

    load_check_support!(Val{np}(), ext_type, support_set, nt, nx, zero(MC{np, NS}))
    if last_obj !== nothing
        if last_obj.integrator === nothing
            ext_type.obj = SupportedFunction(last_obj, support_set.s, last_obj.params, integrator)
        else
            ext_type.obj = SupportedFunction(last_obj, support_set.s, last_obj.params, last_obj.integrator)
        end
    end
    ext_type.lower_storage_interval.x_set_traj.nt = nt
    ext_type.lower_storage_interval.x_set_traj.nx = nx
    ext_type.lower_storage_relax.x_set_traj.nt = nt
    ext_type.lower_storage_relax.x_set_traj.nx = nx

    for i = 1:nt
        push!(ext_type.lower_storage_interval.x_set_traj.v, zeros(Interval{Float64}, nx))
        push!(ext_type.lower_storage_relax.x_set_traj.v, zeros(MC{np,NS}, nx))
        push!(ext_type.x_traj.v, zeros(Float64, nx))
    end

    # add constraint functions
    for cons in t.cons
        cintegrator = cons.integrator === nothing ? integrator : cons.integrator
        push!(ext_type.cons, SupportedFunction(cons, support_set.s, cons.params, cintegrator))
        push!(ext_type.cons_val, 0.0)
    end
    m._subsolvers.ext = ext_type
    m._presolve_time = time() - m._parse_time

    vi = m._working_problem._variable_info
    @show vi
    n = NodeBB(lower_bound.(vi), upper_bound.(vi), is_integer.(vi))
    m._current_node = n

    return nothing
end


function load_integrator!(integrator, lvbs, uvbs, xref)
    DBB.setall!(integrator, DBB.ParameterBound{Lower}(), lvbs)
    DBB.setall!(integrator, DBB.ParameterBound{Upper}(), uvbs)
    DBB.setall!(integrator, DBB.ParameterValue(), xref)
    return nothing
end

function lower_bound_problem!(::Val{true}, t::DynamicExt, opt::GlobalOptimizer)

    # reset box used to evaluate relaxation
    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds
    integrator = t.integrator

    update_relaxed_problem_box!(opt)

    DBB.setall!(integrator, DBB.ParameterBound{Lower}(), lvbs)
    DBB.setall!(integrator, DBB.ParameterBound{Upper}(), uvbs)

    # set reference point to evaluate relaxation
    @__dot__ opt._current_xref = 0.5*(lvbs + uvbs)
    DBB.setall!(integrator, ParameterValue(), opt._current_xref)
    @__dot__ t.p_intv = Interval(lvbs, uvbs)
    @__dot__ t.lower_storage.p_set = MC{t.np,NS}(opt._current_xref, t.p_intv, 1:t.np)

    # relaxes pODE
    relax!(integrator)

    for i = 1:t.nt
        support_time = t.obj.support[i]
        DBB.get(t.lo[i], integrator, DBB.Bound{Lower}(support_time))
        DBB.get(t.hi[i], integrator, DBB.Bound{Upper}(support_time))
        DBB.get(t.cv[i], integrator, DBB.Relaxation{Lower}(support_time))
        DBB.get(t.cc[i], integrator, DBB.Relaxation{Upper}(support_time))
        DBB.get(t.cv_grad[i], integrator, DBB.Subgradient{Lower}(support_time))
        DBB.get(t.cc_grad[i], integrator, DBB.Subgradient{Upper}(support_time))
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
        saf_temp_cons = MOI.ScalarAffineFunction{Float64}(sat_vec, -cons_mc.cv)
        for j = 1:t.np
            coeff = @inbounds cons_mc.cv_grad[j]
            saf_temp_cons.terms[j] = MOI.ScalarAffineTerm{Float64}(coeff, MOI.VariableIndex(j))
            pv = opt._current_xref[j]
            saf_temp_cons.constant += coeff*pv
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

function objective_relax!(opt, relaxed_optimizer, t::DynamicExt, integrator)

    for i = 1:t.nt
        support_time = t.obj.support[i]
        DBB.get!(t.lo[i], integrator, DBB.Bound{Lower}(DBB.TimeIndex(i)))
        DBB.get!(t.hi[i], integrator, DBB.Bound{Upper}(DBB.TimeIndex(i)))
        DBB.get!(t.cv[i], integrator, DBB.Relaxation{Lower}(DBB.TimeIndex(i)))
        DBB.get!(t.cc[i], integrator, DBB.Relaxation{Upper}(DBB.TimeIndex(i)))
        DBB.get!(t.cv_grad[i], integrator, DBB.Subgradient{Lower}(DBB.TimeIndex(i)))
        DBB.get!(t.cc_grad[i], integrator, DBB.Subgradient{Upper}(DBB.TimeIndex(i)))
    end
    load_intervals!(t.x_intv, t.lo, t.hi, t.nt)
    load_trajectory!(t.lower_storage_relax.x_set_traj, t.cv,
                     t.cc, t.x_intv, t.cv_grad, t.cc_grad)

    t.lower_storage_relax.obj_set = t.obj(t.lower_storage_relax.x_set_traj,
                                          t.lower_storage_relax.p_set)
    saf_temp = opt._working_problem._objective_saf
    saf_temp.constant = t.lower_storage_relax.obj_set.cv
    @show saf_temp
    for i = 1:t.np
        coeff = @inbounds t.lower_storage_relax.obj_set.cv_grad[i]
        saf_temp.terms[i] = MOI.ScalarAffineTerm{Float64}(coeff, MOI.VariableIndex(i))
        pv = opt._current_xref[i]
        saf_temp.constant = saf_temp.constant - coeff*pv
    end
    MOI.set(relaxed_optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf_temp)

    return nothing
end

function objective_relax!(opt::GlobalOptimizer, t::DynamicExt, integrator)
    for i = 1:t.nt
        support_time = t.obj.support[i]
        DBB.get(t.lo[i], integrator, DBB.Bound{Lower}(DBB.TimeIndex(i)))
        DBB.get(t.hi[i], integrator, DBB.Bound{Upper}(DBB.TimeIndex(i)))
    end
    load_intervals!(t.x_intv, t.lo, t.hi, t.nt)
    for i = 1:t.nt
        t.lower_storage_interval.x_set_traj.v[i] .= t.x_intv[i]
    end
    t.lower_storage_interval.obj_set = t.obj(t.lower_storage_interval.x_set_traj,
                                             t.lower_storage_interval.p_set)
    opt._lower_objective_value = lo(t.lower_storage_interval.obj_set)
    return nothing
end

function constraint_relax(opt::GlobalOptimizer, relaxed_optimizer, t::DynamicExt, integrator, j)
    for i = 1:t.nt
        support_time = t.cons[j].support[i]
        DBB.get(t.lo[i], integrator, DBB.Bound{Lower}(DBB.TimeIndex(i)))
        DBB.get(t.hi[i], integrator, DBB.Bound{Upper}(DBB.TimeIndex(i)))
        DBB.get(t.cv[i], integrator, DBB.Relaxation{Lower}(DBB.TimeIndex(i)))
        DBB.get(t.cc[i], integrator, DBB.Relaxation{Upper}(DBB.TimeIndex(i)))
        DBB.get(t.cv_grad[i], integrator, DBB.Subgradient{Lower}(DBB.TimeIndex(i)))
        DBB.get(t.cc_grad[i], integrator, DBB.Subgradient{Upper}(DBB.TimeIndex(i)))
    end
    load_intervals!(t.x_intv, t.lo, t.hi, t.nt)
    load_trajectory!(t.lower_storage_relax.x_set_traj, t.cv, t.cc, t.x_intv, t.cv_grad, t.cc_grad)

    # computes relaxation of constraint
    cons_mc = t.cons[j].f(t.lower_storage_relax.x_set_traj, t.lower_storage_relax.p_set)

    # adds linear affine function
    sat_vec = MOI.ScalarAffineTerm{Float64}[MOI.ScalarAffineTerm{Float64}(0.0, MOI.VariableIndex(i)) for i = 1:t.np]
    saf_temp_cons = MOI.ScalarAffineFunction{Float64}(sat_vec, -cons_mc.cv)
    for i = 1:t.np
        coeff = @inbounds cons_mc.cv_grad[i]
        saf_temp_cons.terms[i] = MOI.ScalarAffineTerm{Float64}(coeff, MOI.VariableIndex(i))
        pv = opt._current_xref[i]
        saf_temp_cons.constant += coeff*pv
    end
    constant_value = saf_temp_cons.constant
    saf_temp_cons.constant = 0.0
    MOI.add_constraint(relaxed_optimizer, saf_temp_cons, MOI.LessThan{Float64}(constant_value))
    return lo(cons_mc)
end

function constraint_relax(t::DynamicExt, integrator, i)
    for j = 1:t.nt
        support_time = t.cons[i].support[j]
        DBB.get(t.lo[j], integrator, DBB.Bound{Lower}(DBB.TimeIndex(i)))
        DBB.get(t.hi[j], integrator, DBB.Bound{Upper}(DBB.TimeIndex(i)))
    end
    load_intervals!(t.x_intv, t.lo, t.hi, t.nt)
    for j = 1:t.nt
        t.lower_storage_interval.x_set_traj.v[j] .= t.x_intv[j]
    end
    return t.cons[i].f(t.lower_storage_interval.x_set_traj, t.lower_storage_interval.p_set).lo
end

function EAGO.lower_problem!(t::DynamicExt, opt::GlobalOptimizer)
    println("ran global lower problem...")
    # reset box used to evaluate relaxation
    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds

    @__dot__ opt._current_xref = 0.5*(lvbs + uvbs)
    @__dot__ t.p_intv = Interval(lvbs, uvbs)
    @__dot__ t.lower_storage_interval.p_set = t.p_intv
    @__dot__ t.lower_storage_relax.p_set = MC{t.np,NS}(opt._current_xref, t.p_intv, 1:t.np)

    d = _relaxed_optimizer(opt)
    update_relaxed_problem_box!(opt)

    feasible = true
    support_aff = false
    for (i,cons) in enumerate(t.cons)
        integrator = cons.integrator
        load_integrator!(integrator, lvbs, uvbs, opt._current_xref)
        DBB.relax!(integrator)
        aff_flag = supports_affine(integrator)
        support_aff = support_aff || aff_flag
        if aff_flag
            cval = constraint_relax(opt, d, t, integrator, i)
        else
            cval = constraint_relax(t, integrator, i)
        end
        if cval > 0.0
            feasible = false
            break
        end
    end

    if feasible
        obj_integrator = t.obj.integrator
        if obj_integrator !== NoIntegrator()
            load_integrator!(obj_integrator, lvbs, uvbs, opt._current_xref)
            DBB.relax!(obj_integrator)
            support_aff_obj = supports_affine(obj_integrator)
            support_aff = support_aff || support_aff_obj
            if support_aff
                objective_relax!(opt, d, t, obj_integrator)
            else
                objective_relax!(opt, t, obj_integrator)
            end
        else
            if support_aff
                t.lower_storage_relax.obj_set = t.obj(t.lower_storage_relax.x_set_traj,
                                                      t.lower_storage_relax.p_set)
            else
                t.lower_storage_interval.obj_set = t.obj(t.lower_storage_interval.x_set_traj,
                                                      t.lower_storage_interval.p_set)
            end
        end

        if support_aff
            MOI.optimize!(d)
            t_status = MOI.get(d, MOI.TerminationStatus())
            @show t_status
            p_status = MOI.get(d, MOI.PrimalStatus())
            @show p_status
            d_status = MOI.get(d, MOI.DualStatus())
            @show d_status
            opt._lower_termination_status = t_status
            opt._lower_primal_status = p_status
            opt._lower_dual_status = d_status
            status = EAGO.relaxed_problem_status(t_status, p_status, d_status)
            if status == EAGO.RRS_INFEASIBLE
                opt._lower_feasibility  = false
                opt._lower_objective_value = -Inf
                return
            end
        
            # set dual values
            EAGO.set_dual!(opt)
            opt._lower_feasibility = true
            EAGO.store_lower_solution!(opt, d)
            if status == EAGO.RRS_DUAL_FEASIBLE
                opt._lower_objective_value = MOI.get(d, MOI.DualObjectiveValue())
            end
        end
    else
        opt._lower_objective_value = -Inf
        opt._lower_feasibility = false
    end

    if support_aff
        obj_intv = t.lower_storage_relax.obj_set.Intv
    else
        obj_intv = t.lower_storage_interval.obj_set
    end
    opt._lower_objective_value = max(opt._lower_objective_value, obj_intv.lo)
    return nothing
end

function set_dual_trajectory!(::Val{NP}, t::AbstractDynamic, integrator) where NP
    DBB.getall!(t.value_temp, integrator, DBB.Value())
    DBB.getall!(t.gradient_temp, integrator, DBB.Gradient{Nominal}())
    temp = zeros(NP)
    out = zeros(Dual{TAG,Float64,NP},t.nx)
    for i = 1:t.nt
        for k = 1:t.nx
            for j = 1:t.np
                temp[j] = t.gradient_temp[j][k,i]
            end
            out[k] = Dual{TAG,Float64,NP}(t.value_temp[k,i], Partials{NP,Float64}(tuple(temp...)))
        end
        t.upper_storage.x_set_traj.v[i] .= out
    end
    return nothing
end

function evaluate_dynamics(::Val{NP}, t, param, p, integrator) where NP
    DBB.setall!(integrator, DBB.ConstantParameterValue(), param)
    new_point = t.p_val != p
    if new_point
        DBB.setall!(integrator, DBB.ParameterValue(), p)
        DBB.integrate!(integrator)
        for i = 1:t.nt
            support_time = t.obj.support[i]
            DBB.get(t.x_val[i], integrator, DBB.Value(support_time))
            t.x_traj.v[i] .= t.x_val[i]
        end
        seeds = construct_seeds(Partials{NP,Float64})
        @__dot__ t.upper_storage.p_set = Dual{TAG,Float64,NP}(p, seeds)
        t.p_val .= p
    end
    return new_point
end

function obj_wrap(::Val{NP}, t::AbstractDynamic, p) where NP
    new_eval = evaluate_dynamics(Val{NP}(),t, t.obj.params, p, t.obj.integrator)
    t.obj_val = t.obj.f(t.x_traj, t.p_val)
    return t.obj_val
end

function obj_wrap(t::AbstractDynamic, p)
    t.scalar_temp[1] = p
    new_eval = evaluate_dynamics(Val{1}(),t, t.obj.params,t.scalar_temp, t.obj.integrator)
    t.obj_val = t.obj.f(t.x_traj, t.p_val)
    return t.obj_val
end

function cons_wrap(::Val{NP}, t::AbstractDynamic, i, p) where NP
    new_eval = evaluate_dynamics(Val{NP}(), t, t.cons[i].params, p, t.cons[i].integrator)
    t.cons_val[i] = t.cons[i].f(t.x_traj, t.p_val)
    return t.cons_val[i]
end

function cons_wrap(t::AbstractDynamic, i, p)
    t.scalar_temp[1] = p
    new_eval = evaluate_dynamics(Val{1}(), t, t.cons[i].params, t.scalar_temp, t.cons[i].integrator)
    t.cons_val[i] = t.cons[i].f(t.x_traj, t.p_val)
    return t.cons_val[i][1]
end

function ∇obj_wrap!(::Val{NP}, t::AbstractDynamic, out, p) where NP
    new_eval = evaluate_dynamics(Val{NP}(),t, t.obj.params, p, t.obj.integrator)
    set_dual_trajectory!(Val{NP}(),t, t.obj.integrator)
    obj_dual = t.obj.f(t.upper_storage.x_set_traj, t.upper_storage.p_set)
    out .= partials(obj_dual)
    return nothing
end

function ∇obj_wrap(t::AbstractDynamic, p)
    t.scalar_temp[1] = p
    new_eval = evaluate_dynamics(Val{1}(),t, t.obj.params, t.scalar_temp, t.obj.integrator)
    set_dual_trajectory!(Val{1}(),t, t.obj.integrator)
    obj_dual = t.obj.f(t.upper_storage.x_set_traj, t.upper_storage.p_set)
    return partials(obj_dual, 1)
end

function ∇cons_wrap!(::Val{NP}, t::AbstractDynamic, out, i, p) where NP
    new_eval = evaluate_dynamics(Val{NP}(), t, t.cons[i].params, p, t.cons[i].integrator)
    set_dual_trajectory!(Val{NP}(),t, t.cons[i].integrator)
    cons_dual = t.cons[i].f(t.upper_storage.x_set_traj, t.upper_storage.p_set)
    out .= partials(cons_dual)
    return nothing
end

function ∇cons_wrap(t::AbstractDynamic, i, p)
    t.scalar_temp[1] = p
    new_eval = evaluate_dynamics(Val{1}(), t, t.cons[i].params, t.scalar_temp, t.cons[i].integrator)
    set_dual_trajectory!(Val{1}(),t, t.cons[i].integrator)
    cons_dual = t.cons[i].f(t.upper_storage.x_set_traj, t.upper_storage.p_set)
    return partials(cons_dual, 1)
end

function upper_problem_obj_only!(t::AbstractDynamic, opt::GlobalOptimizer, lvbs, uvbs)
    # get all at particular points???
    DBB.set!(t.integrator, DBB.LocalSensitivityOn(), false)

    obj_integrator = t.obj.integrator
    @__dot__ t.p_val = 0.5*(lvbs + uvbs)
    if obj_integrator !== NoIntegrator()
        DBB.integrate!(obj_integrator)
        for i = 1:t.nt
            support_time = t.obj.support[i]
            DBB.get(t.x_val[i], obj_integrator, DBB.Value(support_time))
            t.x_traj.v[i] .= t.x_val[i]
        end
    end
    t.obj_val = t.obj.f(t.x_traj, t.p_val)
    opt._upper_objective_value = t.obj_val
    opt._upper_feasibility = true
    @__dot__ opt._upper_solution = t.p_val

    return nothing
end

function EAGO.upper_problem!(q::AbstractDynamic, opt::GlobalOptimizer)
    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds
    isempty(q.cons) && return upper_problem_obj_only!(q, opt, lvbs, uvbs)

    np = q.np
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    set_optimizer_attribute(model, "print_level", 1)
    @variable(model, lvbs[i] <= p[i=1:q.np] <= uvbs[i])

    DBB.set!(q.integrator, DBB.LocalSensitivityOn(), true) # TODO: FIX THIS...

    # define the objective
    if isone(q.np)
        JuMP.register(model, :obj, 1, p -> obj_wrap(deepcopy(q), p),
                                      p -> ∇obj_wrap(deepcopy(q), p),
                                      p -> error("Hessian called but is currently disabled."))

        nl_obj = :(obj($(p[1])))
    else
        JuMP.register(model, :obj, np, (p...) -> obj_wrap(Val{np}(), deepcopy(q), p),
                                       (out, p...) -> ∇obj_wrap!(Val{np}(), deepcopy(q), out, p),
                                       (out, p...) -> error("Hessian called but is currently disabled."))
        nl_obj = Expr(:call)
        push!(nl_obj.args, :obj)
        for i in 1:q.np
            push!(nl_obj.args, p[i])
        end
    end
    set_NL_objective(model, MOI.MIN_SENSE, nl_obj)

    for (i,cons) in enumerate(q.cons)
        cons_udf_sym = Symbol("cons_udf_$i")
        if isone(q.np)
            JuMP.register(model, cons_udf_sym, 1, p -> cons_wrap(deepcopy(q), i, p),
                                                  p -> ∇cons_wrap(deepcopy(q), i, p),
                                                  p -> error("Hessian called but is currently disabled."))

            JuMP.add_NL_constraint(model, :(($cons_udf_sym)($(p[1])) <= 0))
        else
            JuMP.register(model, cons_udf_sym, np,
                                 (p...) -> cons_wrap(Val{np}(), deepcopy(q), i, p),
                                 (out, p...) -> ∇cons_wrap!(Val{np}(),deepcopy(q),i,out,p),
                                 (out, p...) -> error("Hessian called but is currently disabled."))

            gic = Expr(:call)
            push!(gic.args, cons_udf_sym)
            for i in 1:q.np
                push!(gic.args, p[i])
            end
            JuMP.add_NL_constraint(model, :($gic <= 0))
        end
    end

    JuMP.optimize!(model)
    t_status = JuMP.termination_status(model)
    r_status = JuMP.primal_status(model)
    feas = EAGO.is_feasible_solution(t_status, r_status)

    if feas
        EAGO.stored_adjusted_upper_bound!(opt, objective_value(model))
    else
        opt._upper_objective_value = Inf
    end
    opt._upper_feasibility = feas
    @__dot__ opt._upper_solution = value(p)

    return nothing
end

EAGO.preprocess!(t::AbstractDynamic, p::GlobalOptimizer) = (p._preprocess_feasibility = true;)
EAGO.postprocess!(t::AbstractDynamic, p::GlobalOptimizer) = (p._postprocess_feasibility = true;)
EAGO.cut_condition(t::AbstractDynamic, p::GlobalOptimizer) = false

function EAGO.optimize_hook!(t::DynamicExt, m::Optimizer)
    println("ran optimize hook dynamic ext") 
    initial_parse!(m)
    optimize!(MINCVX(), m)
end