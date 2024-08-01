cd(@__DIR__)

using DynamicalSystemsBase
using StochasticDiffEq # to enable extention
# We need this because Documenter doesn't know where to get the docstring from otherwise
StochasticSystemsBase = Base.get_extension(DynamicalSystemsBase, :StochasticSystemsBase)

pages = [
    "index.md",
    "CoupledSDEs.md",
]
using DynamicalSystemsBase.SciMLBase

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

build_docs_with_style(pages,
    DynamicalSystemsBase, SciMLBase, StateSpaceSets, StochasticSystemsBase;
)
