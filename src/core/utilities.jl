stateeltype(ds::DynamicalSystem) = typeof(current_state(ds))
StateSpaceSets.dimension(ds::DynamicalSystem) = length(current_state(ds))

safe_state_type(::Val{true}, u0) = ismutable(u0) ? u0: Array(u0)
safe_state_type(::Val{false}, u0) = SVector{length(u0)}(u0...)
safe_state_type(::Val{false}, u0::SVector) = u0
safe_state_type(::Val{false}, u0::Number) = u0

safe_matrix_type(::Val{true}, Q::Matrix) = Q
safe_matrix_type(::Val{true}, Q::AbstractMatrix) = Matrix(Q)
function safe_matrix_type(::Val{false}, Q::AbstractMatrix)
    A, B = size(Q)
    SMatrix{A, B}(Q)
end
save_matrix_type(::Val{false}, Q::SMatrix) = Q
safe_matrix_type(_, a::Number) = a