"""Lazy Julia session manager.

Julia is only initialised when ``WFDRONE_OPT_BACKEND=julia`` (the default)
**and** the first call through ``Main`` is made.  When the backend is
``python``, no Julia process is spawned and importing this module is
essentially free.
"""

import os

_julia_initialized = False
_julia_session = None
_Main = None


def _backend_is_julia() -> bool:
    return os.environ.get("WFDRONE_OPT_BACKEND", "julia").lower() == "julia"


def initialize_julia_session():
    """Initialise Julia session on first call; reuse on subsequent calls."""
    global _julia_initialized, _julia_session, _Main

    if _julia_initialized:
        return _julia_session, _Main

    print("Initializing the Julia session. This can take up to 1 minute.")

    from julia.api import Julia
    _julia_session = Julia(compiled_modules=False)

    from julia import Main
    _Main = Main

    Main.eval("""
    using Logging
    global_logger(SimpleLogger(stderr, Logging.Error))
    """)

    print("initializing the ground sensor julia module")
    Main.include("julia/ground_charging_opt.jl")

    print("initializing the drone julia module")
    Main.include("julia/drone_routing_opt.jl")

    print("initializing the TOP julia module")
    Main.include("julia/TOP.jl")

    print("Julia session initialized.")
    _julia_initialized = True

    return _julia_session, _Main


def get_julia_session():
    """Get the Julia session, initialising it if necessary."""
    if not _julia_initialized:
        return initialize_julia_session()
    return _julia_session, _Main


def reset_julia_session():
    """Reset the Julia session (useful for debugging)."""
    global _julia_initialized, _julia_session, _Main

    if _julia_initialized:
        print("Resetting Julia session...")
        if _julia_session is not None:
            try:
                _julia_session.eval("exit()")
            except Exception:
                pass
        _julia_initialized = False
        _julia_session = None
        _Main = None
        print("Julia session reset complete.")


class _LazyMain:
    """Proxy that delays Julia initialisation until an attribute is accessed."""

    def __getattr__(self, name: str):
        if not _julia_initialized:
            initialize_julia_session()
        return getattr(_Main, name)


# Public API – ``Main`` is now lazy; importing this module no longer starts Julia.
Main = _LazyMain()
