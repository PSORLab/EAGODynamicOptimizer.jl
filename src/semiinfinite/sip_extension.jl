
function llp_check_error!(prob, tstatus, pstatus)
    valid_result, is_feasible = is_globally_optimal(tstatus, pstatus)
    if problem.local_solver && ~((tstatus === MOI.LOCALLY_SOLVED) || (tstatus === MOI.ALMOST_LOCALLY_SOLVED))
        error("Lower problem did not solve to local optimality.")
    elseif ~valid_result
        error("Error encountered in lower level problem.
               Termination status is $tstatus.
               Primal status is $pstatus.")
    end
end


function sipRes_llp(xbar::Vector{Float64})

    m, p = EAGODynamicModel()
    add_supported_objective!(m, p -> -cb.gSIP[indx](xbar, p))

    #=
    add_uncertainty_constraint!(m, prob)
    =#

    JuMP.optimize!(m)
    termination_status = JuMP.termination_status(m)
    result_status = JuMP.primal_status(m)
    result.solution_time += MOI.get(m, MOI.SolveTime())
    llp_check_error!(prob, termination_status, result_status)

    return -JuMP.objective_value(m), JuMP.value.(p), is_feasible
end

function explicit_sip_solve(x_l::Vector{Float64}, x_u::Vector{Float64},
                            p_l::Vector{Float64}, p_u::Vector{Float64},
                            f::Function, gSIP::Function; kwargs...)

    explicit_sip_solve(x_l, x_u, p_l, p_u, f, [gSIP], kwargs...)
end


add_supported_objective!(m, obj)
