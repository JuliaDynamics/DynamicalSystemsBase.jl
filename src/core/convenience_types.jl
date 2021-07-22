export DynamicalSystemIntegrator, DynSysInteg

# how the hell do you write this...???
const DynamicalSystemIntegrator{IIP, S} = Union{
    AbstractODEIntegrator{Alg, IIP, S},
    TangentDiscreteIntegrator{IIP, S},
    MinimalDiscreteIntegrator{IIP, S}
} where {Alg}

const DynSysInteg = DynamicalSystemIntegrator
