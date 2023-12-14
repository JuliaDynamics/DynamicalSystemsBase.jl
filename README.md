# DynamicalSystemsBase.jl

[![docsdev](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystemsbase/dev/)
[![docsstable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystemsbase/stable/)
[![](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--91032--7-purple)](https://link.springer.com/book/10.1007/978-3-030-91032-7)
[![CI](https://github.com/JuliaDynamics/DynamicalSystemsBase.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/DynamicalSystemsBase.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaDynamics/DynamicalSystemsBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDynamics/DynamicalSystemsBase.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/DynamicalSystemsBase)](https://pkgs.genieframework.com?packages=DynamicalSystemsBase)

A Julia package that defines the `DynamicalSystem` interface and many concrete implementations used in the DynamicalSystems.jl ecosystem.

To install it, run `import Pkg; Pkg.add("DynamicalSystemsBase")`.
Typically, you do not want to use `DynamicalSystemsBase` directly,
as downstream analysis packages re-export it.

All further information is provided in the documentation, which you can either find online or build locally by running the `docs/make.jl` file.
