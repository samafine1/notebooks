using Piccolo
using LinearAlgebra
using PiccoloQuantumObjects

# Fun exercise: Implement comparison between two types of models of errors in a Hamitonian for robust control


pretty_print(X::AbstractMatrix) = Base.show(stdout, "text/plain", X); # Helper function


# set time parameters
T = 50
Δt = 0.2
H_drive = PAULIS.X
U_goal = GATES.X
rob_scale = 1 / 8.0
ϵ = 0
piccolo_opts = PiccoloOptions(verbose=false)

# multiplicative error system


# Additive error system



# error sensitivity comparison

# For multiplicative error

# For additive error
add_rob_n = norm(rob_scale * prob_add_rob.trajectory.Ũ⃗ᵥ1) |> println
