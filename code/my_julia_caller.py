# 1) PyJulia / Julia imports
from julia.api import Julia

Julia(compiled_modules=False)  # ensures a "safe" load
from julia import Main, Base

# Initialize Julia and set up any configurations
Main.eval("""
using Logging
global_logger(SimpleLogger(stderr, Logging.Error))  # Silence info logs
""")
print("Initializing the Julia session. This can take up to 1 minute.")

print("initializing the ground sensor julia module")
Main.include("julia/ground_charging_opt.jl")

print("initializing the drone julia module")
Main.include("julia/drone_routing_opt.jl")

Main.include("julia/drone_routing_opt_linear.jl")

print("Julia session initialized.")
# Now `Main` can be imported and reused in other parts of the program: this creates a unique shared Julia session
