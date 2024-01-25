cd(@__DIR__)

using DynamicalSystemsBase

pages = [
    "index.md",
]
using DynamicalSystemsBase.SciMLBase

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

build_docs_with_style(pages,
    DynamicalSystemsBase, SciMLBase, StateSpaceSets;
)
