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
end
function SupportedFunction(f, support::Vector{Float64})
    return SupportedFunction(f, 1, support, Float64[], false)
end
function SupportedFunction(f, support::Vector{Float64}, params::Vector{Float64})
    return SupportedFunction(f, 1, support, params, true)
end
(d::SupportedFunction)(x, p) = d.f(x, p)

const TAG = :DynamicTag

mutable struct DynamicExt{S,T} <: EAGO.ExtensionType
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
    lo::Vector{Vector{Float64}}
    hi::Vector{Vector{Float64}}
    cv::Vector{Vector{Float64}}
    cc::Vector{Vector{Float64}}
    cv_grad::Vector{Matrix{Float64}}
    cc_grad::Vector{Matrix{Float64}}
    lower_storage::SubStorage{S}
    upper_storage::SubStorage{T}
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
    lower_storage = LowerStorage{S}()
    lower_storage.p_set = zeros(S,np)
    upper_storage = LowerStorage{Dual{TAG,Float64,np}}()
    upper_storage.p_set = zeros(Dual{TAG,Float64,np},np)
    DynamicExt{S,T}(integrator, obj, cons, np, nx, nt, p_val, p_intv, x_val,
                  x_intv, x_traj, obj_val, lo, hi, cv, cc, cv_grad, cc_grad,
                  lower_storage,upper_storage)
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

Base.eltype(::DynamicExt{T}) where T = T

function add_supported_objective!(t::Model, obj)
    ext_type = get_optimizer_attribute(t, "ext_type")
    ext_type.obj = SupportedFunction(obj, Float64[])
    set_optimizer_attribute(t, "ext_type", ext_type)
    return nothing
end

function add_supported_objective!(t::Model, params::Vector{Float64}, obj)
    ext_type = get_optimizer_attribute(t, "ext_type")
    ext_type.obj = SupportedFunction(obj, Float64[], params)
    set_optimizer_attribute(t, "ext_type", ext_type)
    return nothing
end

function add_supported_constraint!(t::Model, cons)
    ext_type = get_optimizer_attribute(t, "ext_type")
    push!(ext_type.cons, SupportedFunction(cons, Float64[]))
    set_optimizer_attribute(t, "ext_type", ext_type)
    return nothing
end

function add_supported_constraint!(t::Model, params::Vector{Float64}, cons)
    ext_type = get_optimizer_attribute(t, "ext_type")
    push!(ext_type.cons, SupportedFunction(cons, Float64[], params))
    set_optimizer_attribute(t, "ext_type", ext_type)
    return nothing
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
