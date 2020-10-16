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

function lower_problem!(t::DynamicExt, opt::EAGO.Optimizer)

    n = opt._current_node

    # TODO: calculate

    opt._lower_objective_value = relax_problem_value
    opt._lower_solution = relax_problem_solution
    opt._lower_feasibility = relaxed_problem_feasibility
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
