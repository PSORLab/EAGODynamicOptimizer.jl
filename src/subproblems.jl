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

mutable struct DynamicExt{T} <: EAGO.ExtensionType
    integrator
    obj::Union{SupportedFunction,Nothing}
    np::Int
    nx::Int
    p_val::Vector{Float64}
    x_val::Vector{Float64}
    obj_val::Float64
    lo::Matrix{Float64}
    hi::Matrix{Float64}
    cv::Matrix{Float64}
    cc::Matrix{Float64}
    cv_grad::Vector{Matrix{Float64}}
    cc_grad::Vector{Matrix{Float64}}
    lower_storage::LowerStorage{T}
end

function DynamicExt(integrator, ::T) where T
    obj = nothing
    np = 0
    nx = 0
    p_val = zeros(1)
    x_val = zeros(1)
    obj_val = 0.0
    lo = zeros(1,1)
    hi = zeros(1,1)
    cv = zeros(1,1)
    cc = zeros(1,1)
    cv_grad = Matrix{Float64}[]
    cc_grad = Matrix{Float64}[]
    lower_storage = LowerStorage{T}()
    DynamicExt{T}(integrator, obj, np, nx, p_val, x_val, obj_val, lo, hi,
               cv, cc, cv_grad, cc_grad, lower_storage)
end

function DynamicExt(integrator)
    if supports_affine_relaxation(integrator)
        np = DBB.get(integrator, DBB.ParameterNumber())
        return DynamicExt(integrator, zero(MC{np,NS}))
    end
    return DynamicExt(integrator, zero(Interval{Float64}))
end

function add_supported_objective!(t::Model, obj)
    ext_type = get_optimizer_attribute(t, "ext_type")
    ext_type.obj = SupportedFunction(obj, Float64[])
    set_optimizer_attribute(t, "ext_type", ext_type)
    return
end

function EAGO.presolve_global!(t::DynamicExt, m::EAGO.Optimizer)

    # create initial node
    # load initial relaxed problem

    branch_variable_count = m._branch_variable_count

    m._current_xref             = fill(0.0, branch_variable_count)
    m._candidate_xref           = fill(0.0, branch_variable_count)
    m._current_objective_xref   = fill(0.0, branch_variable_count)
    m._prior_objective_xref     = fill(0.0, branch_variable_count)
    m._lower_lvd                = fill(0.0, branch_variable_count)
    m._lower_uvd                = fill(0.0, branch_variable_count)

    # populate in full space until local MOI nlp solves support constraint deletion
    # uses input model for local nlp solves... may adjust this if a convincing reason
    # to use a reformulated upper problem presents itself
    m._lower_solution      = zeros(Float64, m._working_problem._variable_count)
    m._cut_solution        = zeros(Float64, m._working_problem._variable_count)
    m._continuous_solution = zeros(Float64, m._working_problem._variable_count)
    m._upper_solution      = zeros(Float64, m._working_problem._variable_count)
    m._upper_variables     = fill(MOI.VariableIndex(-1), m._working_problem._variable_count)

    # add storage for fbbt
    m._lower_fbbt_buffer   = zeros(Float64, m._working_problem._variable_count)
    m._upper_fbbt_buffer   = zeros(Float64, m._working_problem._variable_count)

    # add storage for obbt ( perform obbt on all relaxed variables, potentially)
    m._obbt_working_lower_index = fill(false, branch_variable_count)
    m._obbt_working_upper_index = fill(false, branch_variable_count)
    m._old_low_index            = fill(false, branch_variable_count)
    m._old_upp_index            = fill(false, branch_variable_count)
    m._new_low_index            = fill(false, branch_variable_count)
    m._new_upp_index            = fill(false, branch_variable_count)
    m._lower_indx_diff          = fill(false, branch_variable_count)
    m._upper_indx_diff          = fill(false, branch_variable_count)
    m._obbt_variable_count      = branch_variable_count

    # add storage for objective cut
    wp = m._working_problem
    obj_type = EAGO.SCALAR_AFFINE
    wp._objective_saf.terms = fill(MOI.ScalarAffineTerm{Float64}(0.0, MOI.VariableIndex(1)), branch_variable_count)

    m._presolve_time = time() - m._parse_time

    return nothing
end

function EAGO.lower_problem!(t::DynamicExt, opt::EAGO.Optimizer)

    integrator = t.integrator
    np = t.np
    supports_affine = supports_affine_relaxation(integrator)

    # reset box used to evaluate relaxation
    n = opt._current_node
    lvbs = n.lower_variable_bounds
    uvbs = n.upper_variable_bounds
    setall!(integrator, ParameterBound{Lower}(), lvbs)
    setall!(integrator, ParameterBound{Upper}(), uvbs)

    # set reference point to evaluate relaxation
    if supports_affine
       @__dot__ m._current_xref = 0.5*(lvbs + uvbs)
       setall!(integrator, ParameterValue(), m._current_xref)
       @__dot__ t.p_intv = Interval(lvbs, uvbs)
       @__dot__ t.p_mc = MC{N,NS}(m._current_xref, t.p_intv, 1:t.np)
   end

    # relaxes pODE
    relax!(integator)

    # unpacks bounds, relaxations, and subgradients at specific points
    # and computes objective bound/relaxation...
    getall!(t.lo, integrator, Bound{Lower}())
    getall!(t.hi, integrator, Bound{Upper}())
    load_trajectory!(t.x_intv, t.lo, t.hi)

    if supports_affine
        getall!(t.cv, integrator, Relaxation{Lower}())
        getall!(t.cc, integrator, Relaxation{Upper}())
        getall!(t.cv_grad, integrator, Subgradient{Lower}())
        getall!(t.cc_grad, integrator, Subgradient{Upper}())
        load_trajectory!(t.x_set, t.cv, t.cc, t.intv, t.cv_grad, t.cc_grad)
        t.obj_set = t.objective(t.x_set, t.p_set)

        # add affine relaxation... to opt problem

    else
        t.obj_intv = t.objective(t.x_intv, t.p_intv)
    end

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
        opt._lower_solution = m._current_xref
        opt._lower_feasibility = true
    end

    return nothing
end

function EAGO.upper_problem!(t::DynamicExt, opt::EAGO.Optimizer)

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
    p._preprocess_feasibility = true
    return nothing
end

function EAGO.postprocess!(t::DynamicExt, p::Optimizer)
    p._postprocess_feasibility = true
    return nothing
end

EAGO.cut_condition(t::DynamicExt, p::Optimizer) = true
