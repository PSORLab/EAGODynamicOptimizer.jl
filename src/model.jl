function EAGODynamicModel(ext::DynamicExt, kwargs...)

    # get bounds on decision variable
    pL = DBB.getall(ext.integrator, DBB.ParameterBound{Lower}())
    pU = DBB.getall(ext.integrator, DBB.ParameterBound{Upper}())
    np = DBB.get(ext.integrator, DBB.ParameterNumber())

    # initialize model and variables
    m = Model(optimizer_with_attributes(EAGO.Optimizer, kwargs...))
    set_optimizer_attribute(m, "ext_type", ext)

    p = @variable(m, pL[i] <= p[i=1:np] <= pU[i])

    return m, p
end
