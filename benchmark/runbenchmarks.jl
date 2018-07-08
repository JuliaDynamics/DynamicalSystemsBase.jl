# Run this in the REPL, not Juno!!!
using PkgBenchmark
bresult = benchmarkpkg("DynamicalSystemsBase"; retune = true)
