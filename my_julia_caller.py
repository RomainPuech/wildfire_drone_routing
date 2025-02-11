from juliacall import Main as jl

# Initialize Julia and set up any configurations
# jl.seval("""
# using Logging
# global_logger(SimpleLogger(stderr, Logging.Error))  # Silence info logs
# """)

jl.include("julia/ground_charging_opt.jl")

# Now `jl` can be imported and reused in other parts of the program: this creates A UNIQUE SHARED JULIA SESSION
