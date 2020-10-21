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
    for i = 1:np
        push!(cv_grad, zeros(nx, nt))
        push!(cc_grad, zeros(nx, nt))
    end
    lower_storage = LowerStorage{T}()
    DynamicExt{T}(integrator, obj, np, nx, nt, p_val, p_intv, x_val,
                  x_intv, obj_val, lo, hi, cv, cc, cv_grad, cc_grad,
                  lower_storage)
end

function DynamicExt(integrator)
    if supports_affine_relaxation(integrator)
        np = DBB.get(integrator, DBB.ParameterNumber())
        nx = DBB.get(integrator, DBB.StateNumber())
        nt = DBB.get(integrator, DBB.SupportNumber())
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

function load_check_support!(t::DynamicExt, m::EAGO.Optimizer,
                             support_set::DBB.SupportSet, nt::Int,
                             nx::Int, ::T) where T
    f = m.ext_type.obj.f
    m.ext_type.obj = SupportedFunction(f, support_set.s)
    for (i,tval) in enumerate(support_set.s)
        t.lower_storage.x_set_traj.time_dict[tval] = i
    end
    for i = 1:nt
        fill!(t.lower_storage.x_set_traj.v, zeros(T, nx))
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
    @show nt
    if supports_affine_relaxation(integrator)
        m.ext_type = DynamicExt(integrator, np, nx, nt, zero(MC{np, NS}))
        m.ext_type.obj = last_obj
        load_check_support!(t, m, support_set, nt, nx,  zero(MC{np, NS}))
    else
        m.ext_type = DynamicExt(integrator, np, nx, nt, zero(Interval{Float64}))
        m.ext_type.obj = last_obj
        load_check_support!(t, m, support_set, nt, nx, zero(Interval{Float64}))
    end

    m._presolve_time = time() - m._parse_time

    return nothing
end

function EAGO.lower_problem!(t::DynamicExt, opt::EAGO.Optimizer)
    @show "ran lower bound"

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
    if supports_affine
       @__dot__ opt._current_xref = 0.5*(lvbs + uvbs)
       setall!(integrator, ParameterValue(), opt._current_xref)
       @__dot__ t.p_intv = Interval(lvbs, uvbs)
       @__dot__ t.lower_storage.p_set = MC{np,NS}(opt._current_xref, t.p_intv, 1:np)
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
        @show "supports affine branch"
        load_trajectory!(t.lower_storage.x_set_traj, t.cv,
                         t.cc, t.x_intv, t.cv_grad, t.cc_grad)
    else
        @show "interval branch"
        for i = 1:nt
            t.lower_storage.x_set_traj.v[i] .= t.x_intv[i]
        end
    end

    # computes objective
    @show t.lower_storage.x_set_traj.v # TODO: FIX x_set_traj loading...
    t.obj_set = t.obj(t.lower_storage.x_set_traj, t.lower_storage.p_set)

    # unpacks objective result to compute lower bound
    if supports_affine
        if valid_flag
            opt._lower_objective_value = MOI.get(relaxed_optimizer, MOI.ObjectiveValue())
            for i = 1:m._working_problem._variable_count
                opt._lower_solution[i] = MOI.get(relaxed_optimizer, MOI.VariablePrimal(), opt._relaxed_variable_index[i])
            end
            #opt._lower_feasibility = supports_affine ? : true
        else
        end
    else
        opt._lower_objective_value = t.obj_intv.lo
        opt._lower_solution = opt._current_xref
        opt._lower_feasibility = true
    end

    return nothing
end

function EAGO.upper_problem!(t::DynamicExt, opt::EAGO.Optimizer)

    @show "ran upper bound"

    # get all at particular points???
    integrate!(t.integator)
    getall!(t.p_val, integrator, DBB.ParameterValue())
    for i = 1:t.nt
        tval = t.obj.support[i]
        t.x_val[i] .= get(integrator, DBB.Value(TimeIndex(tval)))
    end

    load_trajectory!(t.x_traj, t.x_val)
    t.obj_val = t.obj.f(t.x_traj, t.p_val)

    opt._upper_objective_value = t.obj_val
    opt._upper_feasibility = true
    @__dot__ opt._upper_solution = t.p_val

    return nothing
end

function EAGO.preprocess!(t::DynamicExt, p::Optimizer)
    @show "ran preprocess"

    p._preprocess_feasibility = true
    return nothing
end

function EAGO.postprocess!(t::DynamicExt, p::Optimizer)
    p._postprocess_feasibility = true
    return nothing
end

EAGO.cut_condition(t::DynamicExt, p::Optimizer) = true
