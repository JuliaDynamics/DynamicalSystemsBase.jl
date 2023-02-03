stateeltype(ds::DynamicalSystem) = typeof(current_state(ds))

# State conversion for IIP
correct_state(::Val{false}, u0) = SVector{length(u0)}(u0...)
correct_state(::Val{false}, u0::StateSpaceSets.StaticArraysCore.SArray) = u0
correct_state(::Val{true}, u0::AbstractArray{<:Real}) = ismutable(u0) ? u0 : Array(u0)
correct_state(::Val{false}, u0::SVector) = u0
correct_state(::Val, u0::AbstractVector{<:AbstractArray}) = u0 # for skipping parallel

# Norms for ODE integration
@inline vectornorm(u::AbstractVector, t = 0) = @inbounds standardnorm(u[1], t)
@inline vectornorm(u::Real, t = 0) = abs(u)
@inline standardnorm(u::AbstractArray{<:Number}, t = 0) = sqrt(sum(abs2, u))/length(u)
@inline standardnorm(u::Real, t = 0) = abs(u)
@inline standardnorm(u::AbstractArray, t = 0) = sum(standardnorm, u)/length(u)
@inline function matrixnorm(u::AbstractMatrix, t)
    @inbounds x = abs2(u[1,1])
    for i in 2:size(u, 1)
        @inbounds x += abs2(u[i, 1])
    end
    return sqrt(x)/size(u, 1)
end
@inline matrixnorm(u::Real, t) = abs(u)
