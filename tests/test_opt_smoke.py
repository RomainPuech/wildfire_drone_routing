"""Smoke tests for Python HiGHS optimization backends."""
import os
import sys
import tempfile

import numpy as np

# Ensure code/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

os.environ.setdefault("WFDRONE_OPT_BACKEND", "python")


def _tiny_burnmap(path, T=5, N=8, M=8):
    arr = np.zeros((T, N, M), dtype=np.float64)
    arr[:, 2:6, 2:6] = 0.5
    arr[:, 4, 4] = 1.0
    np.save(path, arr)
    return path


def test_sensor_maxcov_solves():
    import opt as pyopt

    with tempfile.TemporaryDirectory() as d:
        f = _tiny_burnmap(os.path.join(d, "bm.npy"))
        grounds, chargers = pyopt.sensor_maxcov_strategy(f, n_grounds=2, n_charging=1)
        assert len(grounds) <= 2
        assert len(chargers) <= 1
        assert len(grounds) + len(chargers) >= 1


def test_routing_init_solves():
    import opt as pyopt

    with tempfile.TemporaryDirectory() as d:
        f = _tiny_burnmap(os.path.join(d, "bm.npy"), T=12, N=6, M=6)
        # place chargers/grounds near center
        cs = [(3, 3)]
        gs = [(2, 2)]
        rm = pyopt.create_routing_model(f, n_drones=1, charging_stations=cs, ground_stations=gs,
                                        optimization_horizon=4, max_battery_time=4)
        plan = pyopt.solve_init_routing(rm, reevaluation_step=2)
        assert plan is not None
        assert len(plan) == 2
        assert len(plan[0]) == 1


if __name__ == "__main__":
    test_sensor_maxcov_solves()
    print("sensor_maxcov OK")
    test_routing_init_solves()
    print("routing_init OK")
