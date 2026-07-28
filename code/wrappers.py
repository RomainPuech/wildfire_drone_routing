import importlib.util
import os
import tqdm
import json
from typing import Any, Dict, Tuple, List

# === Sensor Wrapping ===

class LoggableSensorStrategyWrapper:
    def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
        # Get path to burnmap or static map
        burnmap_path = custom_initialization_parameters.get("burnmap_filename")
        if burnmap_path is None:
            raise ValueError("Expected 'burnmap_filename' in custom_initialization_parameters")

        # Use the directory containing the burnmap
        base_dir = os.path.dirname(os.path.abspath(burnmap_path))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Build log filename
        layout_name = custom_initialization_parameters.get("log_file", "layout")
        strategy_name = _deep_unwrap(self.base_strategy_cls).__name__
        N = automatic_initialization_parameters.get("N", 0)
        M = automatic_initialization_parameters.get("M", 0)
        n_ground = automatic_initialization_parameters.get("n_ground_stations", 0)
        n_charging = automatic_initialization_parameters.get("n_charging_stations", 0)
        burnmap_nickname = custom_initialization_parameters.get("burnmap_filename", "").split("/")[-1].split(".")[0]

    
        
        log_dir = os.path.join(os.path.dirname(custom_initialization_parameters["burnmap_filename"]), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create burnmap string if required
        # bm_string = ""
        # if "burnmap_filename" in custom_initialization_parameters:
        #     bm_string = custom_initialization_parameters["burnmap_filename"].split('/')[-1] + '_'

        # Create the log filename
    
        scenario_name = automatic_initialization_parameters.get("scenario_name", "scenario")
        log_filename = f"{strategy_name}_{burnmap_nickname}_{N}N_{M}M_{n_ground}ground_{n_charging}charge.json"

        log_path = os.path.join(log_dir, log_filename)
        self.log_path = log_path

        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if os.path.exists(log_path):
            print(f"[DEBUG] [LoggableSensorStrategy] Using PRECOMPUTED log at: {log_path}")
            with open(log_path, "r") as f:
                data = json.load(f)
                self.ground_sensor_locations = [tuple(loc) for loc in data["ground_sensor_locations"]]
                self.charging_station_locations = [tuple(loc) for loc in data["charging_station_locations"]]
        else:
            print(f"[DEBUG] [LoggableSensorStrategy] NO log found at: {log_path} — computing new placement with {strategy_name}")
            base = self.base_strategy_cls(automatic_initialization_parameters, custom_initialization_parameters)
            self.ground_sensor_locations, self.charging_station_locations = base.get_locations()
            with open(log_path, "w") as f:
                json.dump({
                    "ground_sensor_locations": self.ground_sensor_locations,
                    "charging_station_locations": self.charging_station_locations
                }, f, indent=2)
            print(f"[DEBUG] [LoggableSensorStrategy] Saved new placement log at: {log_path}")

    def get_locations(self):
        return self.ground_sensor_locations, self.charging_station_locations
    

def make_loggable_sensor_strategy(strategy_cls):
    """
    Return a multiprocessing-safe, log-enabled wrapper around `strategy_cls`.
    """
    name = f"{strategy_cls.__name__}Logged"        # e.g. RandomSensorPlacementStrategyLogged

    Wrapped = type(                                   # dynamic class creation
        name,
        (LoggableSensorStrategyWrapper,),
        {
            'base_strategy_cls': strategy_cls,
            'strategy_name'   : strategy_cls.__name__,
            '__module__'      : __name__,            
        }
    )

    # Register the class in the *module* namespace so that pickle
    # can import it from "wrappers.Wrapped" later.
    globals()[name] = Wrapped

    return Wrapped


def _deep_unwrap(cls):
    """
    Recursively get the underlying base class for correct strategy naming,
    even if wrapped in clustering/logging/etc.
    """
    seen = set()
    while True:
        if hasattr(cls, "base_cls"):
            cls = cls.base_cls
        elif hasattr(cls, "base_strategy_cls"):
            cls = cls.base_strategy_cls
        elif hasattr(cls, "__wrapped__"):
            cls = cls.__wrapped__
        elif hasattr(cls, "__origin__"):
            cls = cls.__origin__
        else:
            break
        if cls in seen:
            break
        seen.add(cls)
    return cls



class LoggableDroneStrategyWrapper:
    """
    A transparent wrapper that adds *disk logging & caching* to any
    DroneRoutingStrategy.  The public interface is identical to the inner
    strategy.

    Log format  (JSON)
    ------------
    {
        "initial_drone_locations": [ ("charge", [x,y]), ... ],
        "actions_history": [
            [ ("move",  [dx,dy]), ... ],   # step 0
            [ ("charge",null), ... ],      # step 1
            ...
        ]
    }
    """
    # Will be injected by `make_loggable_drone_strategy`
    base_strategy_cls = None          # type: ignore

    # ------ constructor --------------------------------------------------
    def __init__(self,
                 automatic_initialization_parameters: Dict[str, Any],
                 custom_initialization_parameters: Dict[str, Any]):

        # -----------------------------------------------------------------
        # build inner strategy
        # -----------------------------------------------------------------
        self._inner = self.base_strategy_cls(automatic_initialization_parameters,
                                             custom_initialization_parameters)

        # -----------------------------------------------------------------
        # build log-file path
        # -----------------------------------------------------------------
        burnmap_file  = custom_initialization_parameters["burnmap_filename"]
        base_dir      = os.path.dirname(os.path.abspath(burnmap_file))
        log_dir       = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        strategy_name = _deep_unwrap(self._inner.__class__).__name__
        print(f"[DEBUG] For log naming, using base strategy: {strategy_name}")
        N, M          = automatic_initialization_parameters["N"], automatic_initialization_parameters["M"]
        n_drones      = automatic_initialization_parameters["n_drones"]
        n_charge      = automatic_initialization_parameters["n_charging_stations"]
        n_ground      = automatic_initialization_parameters["n_ground_stations"]
        burnmap_nickname = custom_initialization_parameters.get("burnmap_filename", "").split("/")[-1].split(".")[0]

        # fingerprint of charging-station layout (sorted, Julia → Python index already)
        layout_fp     = "_".join([f"{x}-{y}" for x, y
                                  in sorted(automatic_initialization_parameters["charging_stations_locations"])])


        log_name =  f"{os.path.basename(os.path.dirname(custom_initialization_parameters['burnmap_filename']))}_" + \
                    f"{strategy_name}_" + \
                    f"{burnmap_nickname}_" + \
                    f"{automatic_initialization_parameters['n_drones']}_drones_" + \
                    f"{automatic_initialization_parameters['n_charging_stations']}_charging_stations_" + \
                    f"{automatic_initialization_parameters['n_ground_stations']}_ground_stations_" + \
                    layout_fp + "_" + \
                    (f"{custom_initialization_parameters['optimization_horizon']}_" if 'optimization_horizon' in custom_initialization_parameters else '') + "_" + \
                    (f"{custom_initialization_parameters['reevaluation_step']}_" if 'reevaluation_step' in custom_initialization_parameters else '') + \
                    (f"{custom_initialization_parameters['regularization_param']}_" if 'regularization_param' in custom_initialization_parameters else 'no_regularization') + \
                    "logged_drone_routing.json"

        self._log_path = os.path.join(
            log_dir,
            log_name
        )

        # -----------------------------------------------------------------
        # load or create log structure
        # -----------------------------------------------------------------
        self._step_counter  = 0
        self._loaded_from_disk = False

        if (not custom_initialization_parameters.get("recompute_logfile", False)
                and os.path.exists(self._log_path)):
            print(f"[DEBUG] [LoggableDroneStrategy] Using PRECOMPUTED log at: {self._log_path}")
            with open(self._log_path, "r") as fp:
                self._log = json.load(fp)
            self._loaded_from_disk = True
            # sanity: convert lists→tuples where handy
            if self._log.get("initial_drone_locations"):
                self._log["initial_drone_locations"] = [
                    (st, tuple(pos)) for st, pos in self._log["initial_drone_locations"]
                ]
        else:
            print(f"[DEBUG] [LoggableDroneStrategy] NO log found at: {self._log_path} — computing new actions and logging.")
            self._log = {
                "initial_drone_locations": None,
                "actions_history": []
            }

    # ---------------------------------------------------------------------
    # public API mirrors the inner strategy
    # ---------------------------------------------------------------------
    def get_initial_drone_locations(self):
        if self._log["initial_drone_locations"] is not None:
            return self._log["initial_drone_locations"]

        raw = self._inner.get_initial_drone_locations()
        normalised = self._normalise_initial(raw)
        self._log["initial_drone_locations"] = normalised
        self._flush()
        return raw  # keep original return-type

    def next_actions(self, automatic_step_parameters, custom_step_parameters):
        # serve from cache if available
        if (self._loaded_from_disk
                and self._step_counter < len(self._log["actions_history"])):
            stored = self._log["actions_history"][self._step_counter]
            self._step_counter += 1
            return self._denormalise_actions(stored)

        # else compute + store
        acts = self._inner.next_actions(automatic_step_parameters, custom_step_parameters)
        self._log["actions_history"].append(self._normalise_actions(acts))
        self._step_counter += 1
        self._flush()
        return acts

    # ---------------------------------------------------------------------
    # attribute delegation – behave exactly like the real strategy
    # ---------------------------------------------------------------------
    def __getattr__(self, item):
        return getattr(self._inner, item)

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _flush(self):
        with open(self._log_path, "w") as fp:
            json.dump(self._log, fp, indent=2)

    # ---------- (de)normalisation ---------------------------------------
    @staticmethod
    def _normalise_initial(raw) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Bring *any* return variant to a single list [(state,(x,y)), ...]
        """
        if isinstance(raw, list):
            if raw and isinstance(raw[0], tuple) and isinstance(raw[0][0], str):
                # already [(state,(x,y))]
                print(f"raw: {raw}")
                return [(st, (int(x), int(y))) for (st, (x, y)) in raw]
            # just [(x,y),...]  – assume start on charger
            return [("charge", (int(x), int(y))) for (x, y) in raw]

        if isinstance(raw, tuple) and len(raw) == 2:
            positions, states = raw
            return [(st, (int(x), int(y))) for (x, y), st in zip(positions, states)]

        raise ValueError("Unexpected initial-location format from strategy")

    @staticmethod
    def _normalise_actions(acts):
        out = []
        for typ, param in acts:
            out.append([typ, None if param is None else list(param)])
        return out

    @staticmethod
    def _denormalise_actions(stored):
        return [(typ, None if param is None else tuple(param)) for typ, param in stored]


# -------------------------------------------------------------------------
# factory
# -------------------------------------------------------------------------
def make_loggable_drone_strategy(strategy_cls):
    """
    Same idea for drone strategies.
    """
    base_name = _deep_unwrap(strategy_cls).__name__
    name = f"{strategy_cls.__name__}"

    Wrapped = type(
        name,
        (LoggableDroneStrategyWrapper,),
        {
            'base_strategy_cls': strategy_cls,
            'strategy_name'   : strategy_cls.__name__,
            '__module__': 'wrappers',
        }
    )

    globals()[name] = Wrapped
    return Wrapped


from Strategy import RandomSensorPlacementStrategy, SensorPlacementMaxCoverageGaussianTime, DroneRoutingUniformCoverageResetStatic, DroneRoutingMaxCoverageResetStatic, RandomDroneRoutingStrategy
from clustering import get_wrapped_clustering_strategy

# Register statically at module load

RandomSensorPlacementStrategyLogged = make_loggable_sensor_strategy(RandomSensorPlacementStrategy)
SensorPlacementMaxCoverageGaussianTimeLogged = make_loggable_sensor_strategy(SensorPlacementMaxCoverageGaussianTime)

ClusteredUniformCoverage = get_wrapped_clustering_strategy(DroneRoutingUniformCoverageResetStatic)
ClusteredMaxCoverage = get_wrapped_clustering_strategy(DroneRoutingMaxCoverageResetStatic)
ClusteredRandomStrategy = get_wrapped_clustering_strategy(RandomDroneRoutingStrategy)

DroneRoutingUniformCoverageResetStaticLogged = make_loggable_drone_strategy(ClusteredUniformCoverage)
DroneRoutingMaxCoverageResetStaticLogged = make_loggable_drone_strategy(ClusteredMaxCoverage)
RandomDroneRoutingStrategyLogged = make_loggable_drone_strategy(ClusteredRandomStrategy)

# register here the strategies you'd like to use in parallell benchmark 
globals()["DroneRoutingUniformCoverageResetStatic"] = DroneRoutingUniformCoverageResetStaticLogged
globals()["DroneRoutingMaxCoverageResetStatic"] = DroneRoutingMaxCoverageResetStaticLogged
# globals()["RandomDroneRoutingStrategy"] = RandomDroneRoutingStrategyLogged
globals()["RandomDroneRoutingStrategy"] = RandomDroneRoutingStrategy # we do not log the random drone routing strategy
