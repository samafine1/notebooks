import Pkg
Pkg.activate(".")
Pkg.develop(path="../../QuantumCollocation.jl")
Pkg.add("Revise")
using Revise
using QuantumCollocation

Hₑ = PAULIS.X

a = annihilate(3)
sys = QuantumSystem([(a + a')/2, (a - a')/(2im)])
U_goal = EmbeddedOperator(GATES[:H], sys)
T = 51
Δt = 0.2


prob = UnitarySmoothPulseProblem(
            sys, U_goal, T, Δt;
            H_err=Hₑ
        )

solve!(prob, max_iter=5, verbose=false, print_level=1)