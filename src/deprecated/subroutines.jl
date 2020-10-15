# TODO: ADD differentible... if necessary
function relax_objective!(t::DifferentialExt{NS,P}, x::EAGO.Optimizer, x0::Vector{Float64}) where P

    # Calculates convex relaxation
    if t.objective !== nothing

        y = x._current_node
        opt = x.relaxed_optimizer

        val = SVector{P}(x0)
        lo = SVector{P}(y.lower_variable_bounds)
        up = SVector{P}(y.upper_variable_bounds)

        fmc = relax(t.objective, val, lo, up, t.relax_tag)

        # Add objective relaxation to model
        df = fmc.cv_grad
        saf_const = fmc.cv
        grad_c = 0.0
        x0_c = 0.0

        @inbounds vi = x._lower_variable_index
        np = x._variable_number
        for i in 1:np
            @inbounds grad_c = df[i]
            @inbounds x0_c = x0[i]
            @inbounds vindx = vi[i]
            saf_const -= x0_c*grad_c
            MOI.modify(opt,  MOI.ObjectiveFunction{SAF}(), SCoefC(vindx, grad_c))
        end
        MOI.modify(opt,  MOI.ObjectiveFunction{SAF}(), SConsC(saf_const))
        MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    end
    return
end

function relax_problem!(t::DifferentialExt{NS,P}, x::Optimizer, v::Vector{Float64}, q::Int64) where P
    y = x._current_node
    opt = x.relaxed_optimizer

    val = SVector{P}(x0)
    lo = SVector{P}(y.lower_variable_bounds)
    up = SVector{P}(y.upper_variable_bounds)

    for constr in t._supported_scalar_leq
        vfmc = relax(constr, val, lo, up, t.relax_tag)
        # set and uses leq storage only
    end
    return
end
