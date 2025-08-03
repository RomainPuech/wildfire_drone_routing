# 1) PyJulia / Julia imports
from julia.api import Julia
import os

# Global variable to track if Julia has been initialized
_julia_initialized = False
_julia_session = None
_Main = None

def initialize_julia_session():
    """
    Initialize Julia session only once. Subsequent calls will reuse the existing session.
    """
    global _julia_initialized, _julia_session, _Main
    
    if _julia_initialized:
        print("Julia session already initialized, reusing existing session.")
        return _julia_session, _Main
    
    print("Initializing the Julia session. This can take up to 1 minute.")
    
    # Initialize Julia
    # if image exists, use it, otherwise use the default
    if os.path.exists("my_precompiled_sysimage.so"):
        _julia_session = Julia(compiled_modules=False, sysimage="my_precompiled_sysimage.so")
    else:
        _julia_session = Julia(compiled_modules=False)
    
    # Import Julia modules
    from julia import Main, Base
    _Main = Main
    
    # Initialize Julia and set up any configurations
    Main.eval("""
    using Logging
    global_logger(SimpleLogger(stderr, Logging.Error))  # Silence info logs
    """)
    
    print("initializing the ground sensor julia module")
    Main.include("julia/ground_charging_opt.jl")
    
    print("initializing the drone julia module")
    Main.include("julia/drone_routing_opt.jl")
    
    Main.include("julia/drone_routing_opt_linear.jl")
    
    print("Julia session initialized.")
    
    # Mark as initialized
    _julia_initialized = True
    
    return _julia_session, _Main

def get_julia_session():
    """
    Get the Julia session, initializing it if necessary.
    """
    if not _julia_initialized:
        return initialize_julia_session()
    return _julia_session, _Main

def reset_julia_session():
    """
    Reset the Julia session (useful for debugging or when you need a fresh start).
    """
    global _julia_initialized, _julia_session, _Main
    
    if _julia_initialized:
        print("Resetting Julia session...")
        # Close the existing session if possible
        if _julia_session is not None:
            try:
                _julia_session.eval("exit()")
            except:
                pass  # Ignore errors when closing
        
        _julia_initialized = False
        _julia_session = None
        _Main = None
        print("Julia session reset complete.")

# Initialize Julia session when this module is imported
# This ensures Julia is ready when the module is loaded
_julia_session, Main = initialize_julia_session()

# Now `Main` can be imported and reused in other parts of the program: this creates a unique shared Julia session
