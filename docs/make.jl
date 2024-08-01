cd(@__DIR__)

using DynamicalSystemsBase
using StochasticDiffEq # for extention
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
