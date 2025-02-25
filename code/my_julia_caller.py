from juliacall import Main as jl

# Initialize Julia and set up any configurations
jl.seval("""
using Logging
global_logger(SimpleLogger(stderr, Logging.Error))  # Silence info logs
""")
print("Initializing the Julia session. This can take up to 1 minute.")

print("initializing the ground sensor julia module")
jl.include("julia/ground_charging_opt.jl")

print("initializing the drone julia module")
jl.include("julia/drone_routing_opt.jl")



print("Julia session initialized.")
# Now `jl` can be imported and reused in other parts of the program: this creates A UNIQUE SHARED JULIA SESSION
