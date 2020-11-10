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

mutable struct LowerStorage{T}
    p_set::Vector{T}
    x_set::Matrix{T}
    x_set_traj::Trajectory{T}
    obj_set::T
end
function LowerStorage{T}() where T
    LowerStorage{T}(zeros(T,1), zeros(T,1,1), Trajectory{T}(), zero(T))
end

struct SupportedFunction
    f
    support::Vector{Float64}
end
(d::SupportedFunction)(x, p) = d.f(x, p)

mutable struct DynamicExt{T} <: EAGO.ExtensionType
    integrator
    obj::Union{SupportedFunction,Nothing}
    np::Int
    nx::Int
    nt::Int
    p_val::Vector{Float64}
    p_intv::Vector{Interval{Float64}}
    x_val::Vector{Vector{Float64}}
    x_intv::Vector{Vector{Interval{Float64}}}
    x_traj::Trajectory{Float64}
    obj_val::Float64
    lo::Vector{Vector{Float64}}
    hi::Vector{Vector{Float64}}
    cv::Vector{Vector{Float64}}
    cc::Vector{Vector{Float64}}
    cv_grad::Vector{Matrix{Float64}}
    cc_grad::Vector{Matrix{Float64}}
    lower_storage::LowerStorage{T}
end

function DynamicExt(integrator, np::Int, nx::Int, nt::Int, ::T) where T
    obj = nothing
    p_val = zeros(np)
    p_intv = zeros(Interval{Float64}, np)
    obj_val = 0.0
    x_val = Vector{Float64}[]
    x_intv = Vector{Interval{Float64}}[]
    x_traj = Trajectory{Float64}()
    lo = Vector{Float64}[]
    hi = Vector{Float64}[]
    cv = Vector{Float64}[]
    cc = Vector{Float64}[]
    for i = 1:nt
        push!(x_val, zeros(nx))
        push!(x_intv, zeros(Interval{Float64},nx))
        push!(lo, zeros(nx))
        push!(hi, zeros(nx))
        push!(cv, zeros(nx))
        push!(cc, zeros(nx))
    end
    cv_grad = Matrix{Float64}[]
    cc_grad = Matrix{Float64}[]
    for i = 1:nt
        push!(cv_grad, zeros(nx, np))
        push!(cc_grad, zeros(nx, np))
    end
    lower_storage = LowerStorage{T}()
    lower_storage.p_set = zeros(T,np)
    DynamicExt{T}(integrator, obj, np, nx, nt, p_val, p_intv, x_val,
                  x_intv, x_traj, obj_val, lo, hi, cv, cc, cv_grad, cc_grad,
                  lower_storage)
end

function DynamicExt(integrator)
    np = DBB.get(integrator, DBB.ParameterNumber())
    nx = DBB.get(integrator, DBB.StateNumber())
    nt = DBB.get(integrator, DBB.SupportNumber())
    if supports_affine_relaxation(integrator)
        return DynamicExt(integrator, np, nx, nt, zero(MC{np,NS}))
    end
    return DynamicExt(integrator, np, nx, nt, zero(Interval{Float64}))
end

function add_supported_objective!(t::Model, obj)
    ext_type = get_optimizer_attribute(t, "ext_type")
    ext_type.obj = SupportedFunction(obj, Float64[])
    set_optimizer_attribute(t, "ext_type", ext_type)
    return
end

function load_check_support!(q::DynamicExt, m::EAGO.Optimizer,
                             support_set::DBB.SupportSet, nt::Int,
                             nx::Int, ::T) where T
    t = m.ext_type
    f = t.obj.f
    t.obj = SupportedFunction(f, support_set.s)
    for (i, tval) in enumerate(support_set.s)
        t.lower_storage.x_set_traj.time_dict[tval] = i
        t.x_traj.time_dict[tval] = i
    end
    for i = 1:nt
        fill!(t.lower_storage.x_set_traj.v, zeros(T, nx))
        fill!(t.x_traj.v, zeros(Float64, nx))
    end
    return
end

function load_intervals!(d::Vector{Vector{Interval{Float64}}},
                          xL::Vector{Vector{Float64}},
                          xU::Vector{Vector{Float64}}, nt::Int)
    for i = 1:nt
        @__dot__ d[i] = Interval(xL[i], xU[i])
    end
    return nothing
end

function EAGO.presolve_global!(t::DynamicExt, m::EAGO.Optimizer)

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
    for i = 1:nt
        push!(m.ext_type.x_traj.v, zeros(Float64, nx))
    end
    m.ext_type.lower_storage.x_set_traj.nt = nt
    m.ext_type.lower_storage.x_set_traj.nx = nx

    m._presolve_time = time() - m._parse_time

    return nothing
end

function EAGO.lower_problem!(q::DynamicExt, opt::EAGO.Optimizer)

    t = opt.ext_type
    integrator = t.integrator
    np = t.np
    nt = t.nt
    supports_affine = supports_affine_relaxation(integrator)

    # reset box used to evaluate relaxation
    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds
    setall!(integrator, ParameterBound{Lower}(), lvbs)
    setall!(integrator, ParameterBound{Upper}(), uvbs)

    # set reference point to evaluate relaxation
    @__dot__ opt._current_xref = 0.5*(lvbs + uvbs)
    setall!(integrator, ParameterValue(), opt._current_xref)
    @__dot__ t.p_intv = Interval(lvbs, uvbs)
    if supports_affine
        @__dot__ t.lower_storage.p_set = MC{np,NS}(opt._current_xref, t.p_intv, 1:np)
    else
        @__dot__ t.lower_storage.p_set = t.p_intv
    end


    # relaxes pODE
    relax!(integrator)

    # unpacks bounds, relaxations, and subgradients at specific points
    # and computes objective bound/relaxation...
    for i = 1:nt
        support_time = t.obj.support[i]
        get(t.lo[i], integrator, Bound{Lower}(support_time))
        get(t.hi[i], integrator, Bound{Upper}(support_time))
        if supports_affine
            get(t.cv[i], integrator, Relaxation{Lower}(support_time))
            get(t.cc[i], integrator, Relaxation{Upper}(support_time))
            get(t.cv_grad[i], integrator, Subgradient{Lower}(support_time))
            get(t.cc_grad[i], integrator, Subgradient{Upper}(support_time))
        end
    end
    load_intervals!(t.x_intv, t.lo, t.hi, nt)

    # loads trajectory
    if supports_affine
        load_trajectory!(t.lower_storage.x_set_traj, t.cv,
                         t.cc, t.x_intv, t.cv_grad, t.cc_grad)
    else
        for i = 1:nt
            t.lower_storage.x_set_traj.v[i] .= t.x_intv[i]
        end
    end

    # computes objective
    t.lower_storage.obj_set = t.obj(t.lower_storage.x_set_traj, t.lower_storage.p_set)

    # unpacks objective result to compute lower bound
    if supports_affine

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
        MOI.optimize!(relaxed_optimizer)

        opt._lower_termination_status = MOI.get(relaxed_optimizer, MOI.TerminationStatus())
        opt._lower_result_status = MOI.get(relaxed_optimizer, MOI.PrimalStatus())
        valid_flag, feasible_flag = EAGO.is_globally_optimal(opt._lower_termination_status, opt._lower_result_status)

        if valid_flag
            opt._cut_add_flag = true
            opt._lower_feasibility = true
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
        opt._lower_objective_value = lo(t.lower_storage.obj_set)
        opt._lower_solution = opt._current_xref
        opt._lower_feasibility = true
    end

    return nothing
end

function EAGO.upper_problem!(q::DynamicExt, opt::EAGO.Optimizer)

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

function EAGO.preprocess!(t::DynamicExt, p::Optimizer)
    p._preprocess_feasibility = true
    return nothing
end

function EAGO.postprocess!(t::DynamicExt, p::Optimizer)
    p._postprocess_feasibility = true
    return nothing
end

EAGO.cut_condition(t::DynamicExt, p::Optimizer) = false
