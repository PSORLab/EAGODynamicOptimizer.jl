mutable struct SubStorage{T}
    p_set::Vector{T}
    x_set::Matrix{T}
    x_set_traj::Trajectory{T}
    obj_set::T
end
function SubStorage{T}() where T
    SubStorage{T}(zeros(T,1), zeros(T,1,1), Trajectory{T}(), zero(T))
end

struct SupportedFunction
    f
    n::Int
    support::Vector{Float64}
    params::Vector{Float64}
    has_params::Bool
    integrator
end
function SupportedFunction(f, support::Vector{Float64}, integrator = nothing)
    return SupportedFunction(f, 1, support, Float64[], false, integrator)
end
function SupportedFunction(f, support::Vector{Float64}, params::Vector{Float64}, integrator = nothing)
    return SupportedFunction(f, 1, support, params, true, integrator)
end
(d::SupportedFunction)(x, p) = d.f(x, p)

const TAG = :DynamicTag

# Abstract type created to support EAGO-GPU, which is an extension that wants
# to re-use some parts of EAGODynamicOptimizer. Instead of making DynamicExt
# an extension, it's now a subtype of AbstractDynamic, so it'll perform the
# same way as before, but some functions will work with both DynamicExt
# and the new DynamicExtGPU.
abstract type AbstractDynamic{S,T} <: ExtensionType end

mutable struct DynamicExt{S,T} <: AbstractDynamic{S,T}
    integrator
    obj::Union{SupportedFunction,Nothing}
    cons::Vector{SupportedFunction}
    np::Int
    nx::Int
    nt::Int
    p_val::Vector{Float64}
    p_intv::Vector{Interval{Float64}}
    x_val::Vector{Vector{Float64}}
    x_intv::Vector{Vector{Interval{Float64}}}
    x_traj::Trajectory{Float64}
    obj_val::Float64
    cons_val::Vector{Float64}
    lo::Vector{Vector{Float64}}
    hi::Vector{Vector{Float64}}
    cv::Vector{Vector{Float64}}
    cc::Vector{Vector{Float64}}
    cv_grad::Vector{Matrix{Float64}}
    cc_grad::Vector{Matrix{Float64}}
    lower_storage_interval::SubStorage{Interval{Float64}}
    lower_storage_relax::SubStorage{S}
    upper_storage::SubStorage{T}
    scalar_temp::Vector{Float64}
    value_temp::Matrix{Float64}
    gradient_temp::Vector{Matrix{Float64}}
end

function DynamicExt(integrator, np::Int, nx::Int, nt::Int, ::S) where S
    obj = nothing
    cons = SupportedFunction[]
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
    lower_storage_interval = SubStorage{Interval{Float64}}()
    lower_storage_interval.p_set = zeros(Interval{Float64},np)
    lower_storage_relax = SubStorage{S}()
    lower_storage_relax.p_set = zeros(S,np)
    T = Dual{TAG,Float64,np}
    upper_storage = SubStorage{T}()
    upper_storage.p_set = zeros(T,np)
    scalar_temp = Float64[0.0]
    value_temp = zeros(Float64, nx, nt)
    gradient_temp = Matrix{Float64}[]
    for i = 1:np
        push!(gradient_temp, zeros(nx, nt))
    end
    cons_val = Float64[]
    DynamicExt{S,T}(integrator, obj, cons, np, nx, nt, p_val, p_intv, x_val,
                  x_intv, x_traj, obj_val, cons_val, lo, hi, cv, cc, cv_grad, cc_grad,
                  lower_storage_interval, lower_storage_relax, upper_storage,
                  scalar_temp, value_temp, gradient_temp)
end

function DynamicExt(integrator)
    np = DBB.get(integrator, DBB.ParameterNumber())
    nx = DBB.get(integrator, DBB.StateNumber())
    nt = DBB.get(integrator, DBB.SupportNumber())
    return DynamicExt(integrator, np, nx, nt, zero(MC{np,NS}))
end

Base.eltype(::AbstractDynamic{T}) where T = T

function add_supported_objective!(t::Model, obj)
    ext = get_optimizer_attribute(t, "ext")
    ext.obj = SupportedFunction(obj, Float64[], ext.integrator)
    set_optimizer_attribute(t, "ext", ext)
    return nothing
end

function add_supported_objective!(t::Model, obj, integrator)
    ext = get_optimizer_attribute(t, "ext")
    ext.obj = SupportedFunction(obj, Float64[], integrator)
    set_optimizer_attribute(t, "ext", ext)
    return nothing
end

function add_supported_objective!(t::Model, params::Vector{Float64}, obj)
    ext = get_optimizer_attribute(t, "ext")
    ext.obj = SupportedFunction(obj, Float64[], params, ext.integrator)
    set_optimizer_attribute(t, "ext", ext)
    return nothing
end

function add_supported_objective!(t::Model, params::Vector{Float64}, obj, integrator)
    ext = get_optimizer_attribute(t, "ext")
    ext.obj = SupportedFunction(obj, Float64[], params, integrator)
    set_optimizer_attribute(t, "ext", ext)
    return nothing
end

function add_supported_constraint!(t::Model, cons)
    ext = get_optimizer_attribute(t, "ext")
    push!(ext.cons, SupportedFunction(cons, Float64[], ext.integrator))
    push!(ext.cons_val, 0.0)
    set_optimizer_attribute(t, "ext", ext)
    return nothing
end

function add_supported_constraint!(t::Model, cons, integrator)
    ext = get_optimizer_attribute(t, "ext")
    push!(ext.cons, SupportedFunction(cons, Float64[], integrator))
    push!(ext.cons_val, 0.0)
    set_optimizer_attribute(t, "ext", ext)
    return nothing
end

function load_check_support!(::Val{NP}, t::AbstractDynamic, support_set::DBB.SupportSet,
                             nt::Int, nx::Int, ::T) where {NP,T}
    for (i, tval) in enumerate(support_set.s)
        t.lower_storage_interval.x_set_traj.time_dict[tval] = i
        t.lower_storage_relax.x_set_traj.time_dict[tval] = i
        t.upper_storage.x_set_traj.time_dict[tval] = i
        t.x_traj.time_dict[tval] = i
    end
    for i = 1:nt
        push!(t.lower_storage_interval.x_set_traj.v, zeros(Interval{Float64}, nx))
        push!(t.lower_storage_relax.x_set_traj.v, zeros(T, nx))
        push!(t.upper_storage.x_set_traj.v, zeros(Dual{TAG,Float64,NP}, nx))
        fill!(t.x_traj.v, zeros(Float64, nx))
    end
    return nothing
end

function load_intervals!(d::Vector{Vector{Interval{Float64}}},
                          xL::Vector{Vector{Float64}},
                          xU::Vector{Vector{Float64}}, nt::Int)
    for i = 1:nt
        @__dot__ d[i] = Interval(xL[i], xU[i])
    end
    return nothing
end
