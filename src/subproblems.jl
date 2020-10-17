struct DynamicExt <: EAGO.ExtensionType
    integrator
end

function presolve_global!(t::DynamicExt, m::EAGO.Optimizer)

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
    m._upper_variables     = fill(VI(-1), m._working_problem._variable_count)

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
    obj_type = wp._objective_type # NEED THIS???
    wp._objective_saf.terms = # TODO: SAF()

    m._presolve_time = time() - m._parse_time

    return
end

function supports_affine_relaxation(integrator)
    supports(integrator, Relaxation{Lower}()) &&
    supports(integrator, Relaxation{Upper}()) &&
    supports(integrator, Subgradient{Lower}()) &&
    supports(integrator, Subgradient{Upper}())
end

function lower_problem!(t::DynamicExt, opt::EAGO.Optimizer)

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
       @__dot__ t.p_mc = MC{N,NS}(m._current_xref, t.p_intv, 1:np)
   end

    # relaxes pODE
    relax!(integator)

    # unpacks bounds, relaxations, and subgradients at specific points
    # and computes objective bound/relaxation...
    getall!(t.lo, integrator, Bound{Lower}())
    getall!(t.hi, integrator, Bound{Upper}())
    @__dot__ t.x_intv = Interval(t.lo, t.hi)

    if supports_affine
        getall!(t.cv, integrator, Relaxation{Lower}())
        getall!(t.cc, integrator, Relaxation{Upper}())
        getall!(t.cv_grad, integrator, Subgradient{Lower}())
        getall!(t.cc_grad, integrator, Subgradient{Upper}())
        @__dot__ t.x_mc = MC{N,NS}(t.cv, t.cc, t.intv, t.cv_grad, t.cc_grad, false)
        t.obj_mc = t.objective(t.x_mc, t.p_mc)

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
            opt._lower_feasibility = supports_affine ? : true
        else
        end
    else
        opt._lower_objective_value = t.obj_intv.lo
        opt._lower_solution = m._current_xref
        opt._lower_feasibility = true
    end

    return nothing
end

function upper_problem!(t::DynamicExt, opt::EAGO.Optimizer)

    n = opt._current_node

    # TODO: calculate

    opt._upper_objective_value = upper_problem_value
    opt._upper_solution = upper_problem_solution
    opt._upper_feasibility = upper_problem_feasibility
end

function EAGO.preprocess!(t::DynamicExt, p::Optimizer)
    p._preprocess_feasibility = true
    return
end

function EAGO.postprocess!(t::DynamicExt, p::Optimizer)
    p._postprocess_feasibility = true
    return
end

EAGO.cut_condition(t::DynamicExt, p::Optimizer) = true
