stateeltype(ds::DynamicalSystem) = typeof(current_state(ds))
StateSpaceSets.dimension(ds::DynamicalSystem) = length(current_state(ds))
