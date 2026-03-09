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

    Tries compiled_modules=True first (fast, ~10s) which works when launched
    via ``python-jl``.  Falls back to compiled_modules=False (~60s) when
    running under a statically-linked Python (e.g. Conda).

    To use the fast path, run your script with:
        python-jl run_benchmark.py
    instead of:
        python run_benchmark.py
    """
    global _julia_initialized, _julia_session, _Main
    
    if _julia_initialized:
        print("Julia session already initialized, reusing existing session.")
        return _julia_session, _Main
    
    # Detect julia binary path — Jupyter kernels may not have /opt/homebrew/bin in PATH
    import shutil
    julia_runtime = shutil.which("julia") or "/opt/homebrew/bin/julia"

    # Try fast path first (compiled_modules=True), fall back to slow path
    try:
        print("Initializing Julia session (compiled_modules=True)...")
        _julia_session = Julia(compiled_modules=True, runtime=julia_runtime)
        print("Julia started with compiled_modules=True (fast path).")
    except Exception:
        print("compiled_modules=True not supported (statically-linked Python).")
        print("Falling back to compiled_modules=False (slower startup).")
        print("TIP: Run with 'python-jl' instead of 'python' for faster startup.")
        _julia_session = Julia(compiled_modules=False, runtime=julia_runtime)
    
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
    
    # Main.include("julia/drone_routing_opt_linear.jl")

    print("initializing the TOP julia module")
    Main.include("julia/TOP.jl")
    
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
