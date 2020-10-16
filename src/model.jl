function EAGODynamicModel(ext::DynamicExt, kwargs...)

    # get bounds on decision variable
    pL = getall(ext.integrator. ParameterBound{Lower}())
    pU = getall(ext.integrator, ParameterBound{Upper}())

    # initialize model and variables
    m = Model(with_optimizer(kwargs...))
    p = @variable(m, pL[i] <= p[i=1:prob.np] <= pU[i])

    return m, p
end
