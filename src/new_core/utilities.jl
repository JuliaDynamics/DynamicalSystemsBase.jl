stateeltype(ds::DynamicalSystem) = typeof(current_state(ds))

correct_state(::Val{false}, u0) = SVector{length(u0)}(u0...)
correct_state(::Val{true}, u0::AbstractArray{<:Real}) = ismutable(u0) ? u0 : Array(u0)
correct_state(::Val{false}, u0::SVector) = u0
correct_state(::Val, u0::AbstractVector{<:AbstractArray}) = u0 # for skipping parallel

safe_matrix_type(::Val{true}, Q::Matrix) = Q
safe_matrix_type(::Val{true}, Q::AbstractMatrix) = Matrix(Q)
function safe_matrix_type(::Val{false}, Q::AbstractMatrix)
    A, B = size(Q)
    SMatrix{A, B}(Q)
end
save_matrix_type(::Val{false}, Q::SMatrix) = Q
safe_matrix_type(_, a::Number) = a