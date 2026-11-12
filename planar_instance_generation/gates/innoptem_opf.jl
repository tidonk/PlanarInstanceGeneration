# Persistent base-case OPF worker using the application's existing DC_OPF entry.
using TopologyOptimizer
using JSON
using JuMP
using Gurobi
using PowerModels
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

# Reactive quantities are undefined in DC solutions. Preserve those as null.
json_safe(x::AbstractDict) = Dict(string(k) => json_safe(v) for (k, v) in x)
json_safe(x::AbstractVector) = json_safe.(x)
json_safe(x::AbstractFloat) = isfinite(x) ? x : nothing
json_safe(x) = x

const ENVIRONMENT = Gurobi.Env()
const GUROBI_RUNTIME_VERSION = let
    major, minor, technical = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    Gurobi.GRBversion(major, minor, technical)
    string(major[], ".", minor[], ".", technical[])
end
println("PIG_OPF " * JSON.json(Dict("ready" => true, "julia" => string(VERSION),
    "powermodels" => string(pkgversion(PowerModels)), "gurobi" => GUROBI_RUNTIME_VERSION)))
flush(stdout)
for line in eachline(stdin)
    request = JSON.parse(line)
    get(request, "stop", false) && break
    try
        result = redirect_stdout(stderr) do
            data = PowerModels.parse_file(request["input"])
            options = [String(key) => value for (key, value) in request["solver_params"]]
            solver = JuMP.optimizer_with_attributes(() -> Gurobi.Optimizer(ENVIRONMENT), options...)
            TopologyOptimizer.solve_application!(data, "DC_OPF", Symbol[], 1, false, solver, solver)
        end
        status = string(result["termination_status"])
        primal = string(get(result, "primal_status", "NO_SOLUTION"))
        feasible = primal == "FEASIBLE_POINT" && isfinite(get(result, "objective", Inf))
        output = Dict("feasible" => feasible, "status" => status,
            "objective" => feasible ? result["objective"] : nothing,
            "optimizer_seconds" => get(result, "solve_time", nothing),
            "solution" => feasible ? result["solution"] : nothing)
        open(request["output"], "w") do stream
            JSON.print(stream, json_safe(output))
        end
        println("PIG_OPF " * JSON.json(Dict("done" => true)))
    catch error
        println("PIG_OPF " * JSON.json(Dict("error" => sprint(showerror, error, catch_backtrace()))))
    end
    flush(stdout)
end
