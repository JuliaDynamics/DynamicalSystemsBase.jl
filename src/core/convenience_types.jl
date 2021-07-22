export DynamicalSystemIntegrator, DynSysInteg
const DynSysInteg = DynamicalSystemIntegrator

const DynamicalSystemIntegrator{IIP, S} = Union{
    AbstractODEIntegrator{Alg, IIP, S},
    TangentDiscreteIntegrator{IIP, S},
    MinimalDiscreteIntegrator{IIP, S}
} where {S}
