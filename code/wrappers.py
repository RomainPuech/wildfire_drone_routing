import importlib.util
import os
import tqdm
import json
from typing import Any, Dict, Tuple, List


def wrap_log_sensor_strategy(input_strat_cls, scenario_level_log: bool = False, log_id=""):
    """
    Wraps a SensorPlacementStrategy to log and reuse previous placements.

    Args:
        input_strat_cls (SensorPlacementStrategy): The input sensor placement strategy class.
        bool: scenario_level_log: If True, the log file will be saved at the scenario level, otherwise it will be saved at the layout level

    Returns:
        WrappedStrategy (SensorPlacementStrategy): A wrapped version that logs and reuses results.
    """

    class WrappedStrategy(input_strat_cls):
        def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
            """
            Initialize the wrapped strategy, logging results or loading if already logged.

            automatic_initialization_parameters: dict 
                    Expected keys:
                        - n_ground_stations
                        - n_charging_stations
                        - N, M (grid size)
                        - scenario_name: name of the scenario
                custom_initialization_parameters: dict
                    Expected keys:
                        - log_file: Path to the log file
                        - burnmap_filename: Path to the burn map used by the Julia optimizer
                        - recompute_logfile: If True, the log file will be recomputed
            """

            n_ground = automatic_initialization_parameters.get("n_ground_stations", 0)
            n_charging = automatic_initialization_parameters.get("n_charging_stations", 0)
            N = automatic_initialization_parameters.get("N", 0)
            M = automatic_initialization_parameters.get("M", 0)
            strategy_name = input_strat_cls.__name__

            # Save logs next to burnmap in "logs" directory
            log_id_str = str(log_id)
            if scenario_level_log:
                log_dir = os.path.join(os.path.dirname(custom_initialization_parameters["burnmap_filename"]), "logs", "scenario_level")
            else:
                log_dir = os.path.join(os.path.dirname(custom_initialization_parameters["burnmap_filename"]), "logs")
            os.makedirs(log_dir, exist_ok=True)

            if scenario_level_log:
                log_path = os.path.join(log_dir, f"{automatic_initialization_parameters['scenario_name']}_{custom_initialization_parameters['burnmap_filename'].split('/')[-1]}_{strategy_name}_{N}N_{M}M_{n_ground}ground_{n_charging}charge{log_id_str}.json")
            else:
                log_path = os.path.join(log_dir, f"{custom_initialization_parameters['burnmap_filename'].split('/')[-1]}_{strategy_name}_{N}N_{M}M_{n_ground}ground_{n_charging}charge{log_id_str}.json")


            self.ground_sensor_locations = []
            self.charging_station_locations = []

            if os.path.exists(log_path) and not custom_initialization_parameters.get("recompute_logfile", False):
                # print(f"[wrap_log_strategy] Loading placement from: {log_path}")
                with open(log_path, "r") as log_file:
                    data = json.load(log_file)

                    # Convert list to tuple
                    self.ground_sensor_locations = [tuple(loc) for loc in data["ground_sensor_locations"]]
                    self.charging_station_locations = [tuple(loc) for loc in data["charging_station_locations"]]
            else:
                # print(f"[wrap_log_strategy] Log not found, running {strategy_name}...")
                # call the parent strategy to compute placements
                super().__init__(automatic_initialization_parameters, custom_initialization_parameters)
                # save the computed locations
                self.ground_sensor_locations, self.charging_station_locations = super().get_locations()

                # log to file
                with open(log_path, "w") as log_file:
                    json.dump({
                        "ground_sensor_locations": self.ground_sensor_locations,
                        "charging_station_locations": self.charging_station_locations
                    }, log_file, indent=2)
                # print(f"[wrap_log_strategy] Placements saved to: {log_path}")

        def get_locations(self):
            return self.ground_sensor_locations, self.charging_station_locations

    return WrappedStrategy


def wrap_log_drone_strategy(input_drone_cls, scenario_level_log: bool = False, log_id=""):
    """
    Wraps a DroneRoutingStrategy to add logging capabilities.
    
    This wrapper:
    1. Logs all drone locations and actions to a JSON file
    2. Loads from the log file if it exists (avoiding re-optimization)
    3. Maintains compatibility with different strategy return formats
    
    Args:
        input_drone_cls (class): A DroneRoutingStrategy class to wrap
        scenario_level_log (bool): If True, the log file will be saved at the scenario level, otherwise it will be saved at the layout level
        log_id (str): Additional identifier to append to the log filename
        
    Returns:
        class: A wrapped version of the input class that adds logging functionality
        
    Notes:
        The wrapped class is compatible with strategies that return either:
        - A single list of (state,(x,y)) from get_initial_drone_locations()
        - A 2-tuple (positions, states) from get_initial_drone_locations()
        
        The log file format is:
        {
            "initial_drone_locations": [[(state,(x,y)), ...]],  # for each cluster
            "actions_history": [
                [(action_type, (x,y)), ...],  # step 0
                [(action_type, (x,y)), ...],  # step 1
                ...
            ]
        }
    """

    import json, os

    class LoggedDroneRoutingStrategy(input_drone_cls):
        def __init__(self, automatic_initialization_parameters, custom_initialization_parameters):
            super().__init__(automatic_initialization_parameters, custom_initialization_parameters)
            self.auto_params = automatic_initialization_parameters
            self.custom_params = custom_initialization_parameters

            # We'll store everything in self.log_data
            # Format:
            # {
            #   "initial_drone_locations": [[(state,(x,y)), (state,(x,y)), ...]], # for cluster i
            #   "actions_history": [
            #       [ (action_type, (x,y)), ...],   # step 0
            #       [ (action_type, (x,y)), ...],   # step 1
            #       ...
            #   ]
            # }
            self.log_data = {
                "initial_drone_locations": None,
                "actions_history": []
            }

            # Build a default log filename if not specified
            # or use the user-provided "log_file"
            if "log_file" in custom_initialization_parameters:
                self.log_file = custom_initialization_parameters["log_file"]
                
            # Build log filename with cluster-specific fingerprint
            # Handle scenario_level_log similar to wrap_log_sensor_strategy
            log_id_str = str(log_id)
            if scenario_level_log:
                log_dir = os.path.join(os.path.dirname(custom_initialization_parameters["burnmap_filename"]), "logs", "scenario_level")
            else:
                log_dir = os.path.join(os.path.dirname(custom_initialization_parameters["burnmap_filename"]), "logs")
            os.makedirs(log_dir, exist_ok=True)

            # Create a fingerprint string based on charging station layout
            charging_stations = automatic_initialization_parameters["charging_stations_locations"]
            layout_fingerprint = "_".join([f"{x}-{y}" for x, y in sorted(charging_stations)])

            # Build full filename
            if scenario_level_log:
                log_name = f"{automatic_initialization_parameters['scenario_name']}_" + \
                        f"{custom_initialization_parameters['burnmap_filename'].split('/')[-1]}_" + \
                        f"{input_drone_cls.strategy_name}_" + \
                        f"{automatic_initialization_parameters['n_drones']}_drones_" + \
                        f"{automatic_initialization_parameters['n_charging_stations']}_charging_stations_" + \
                        f"{automatic_initialization_parameters['n_ground_stations']}_ground_stations_" + \
                        layout_fingerprint + "_" + \
                        (f"{custom_initialization_parameters['optimization_horizon']}_" if 'optimization_horizon' in custom_initialization_parameters else '') + "_" + \
                        (f"{custom_initialization_parameters['reevaluation_step']}_" if 'reevaluation_step' in custom_initialization_parameters else '') + \
                        (f"{custom_initialization_parameters['regularization_param']}_" if 'regularization_param' in custom_initialization_parameters else 'no_regularization') + \
                        f"{log_id_str}" + \
                        "logged_drone_routing.json"
            else:
                log_name = f"{custom_initialization_parameters['burnmap_filename'].split('/')[-1]}_" + \
                        f"{input_drone_cls.strategy_name}_" + \
                        f"{automatic_initialization_parameters['n_drones']}_drones_" + \
                        f"{automatic_initialization_parameters['n_charging_stations']}_charging_stations_" + \
                        f"{automatic_initialization_parameters['n_ground_stations']}_ground_stations_" + \
                        layout_fingerprint + "_" + \
                        (f"{custom_initialization_parameters['optimization_horizon']}_" if 'optimization_horizon' in custom_initialization_parameters else '') + "_" + \
                        (f"{custom_initialization_parameters['reevaluation_step']}_" if 'reevaluation_step' in custom_initialization_parameters else '') + \
                        (f"{custom_initialization_parameters['regularization_param']}_" if 'regularization_param' in custom_initialization_parameters else 'no_regularization') + \
                        f"{log_id_str}" + \
                        "logged_drone_routing.json"

            self.log_file = os.path.join(log_dir, log_name)

            self.loaded_from_log = False

            # If user wants to force recomputation, we skip loading
            # Otherwise we try to load from self.log_file
            if not custom_initialization_parameters.get("recompute_logfile", False):
                # print(f"\033[91m WE TRY TO LOAD FROM LOGFILE \033[0m")
                if os.path.exists(self.log_file):
                    # print(f"[wrap_log_drone_strategy] ✅ Log found at {self.log_file}, loading from disk.")
                    with open(self.log_file, "r") as f:
                        data = json.load(f)
                    self.log_data = data
                    self.loaded_from_log = True
                    # print(f"[wrap_log_drone_strategy] Loaded {len(self.log_data.get('actions_history', []))} steps of actions.")
                # else:
                    # print(f"[wrap_log_drone_strategy] 🚫 No log file found at {self.log_file}. Logging will be enabled.")
            # else:
                # print(f"[wrap_log_drone_strategy] 🔄 Forcing recomputation. Will overwrite {self.log_file}.")


            # We'll keep a step counter for next_actions
            self.step_counter = 0

        def get_initial_drone_locations(self):
            """
            If loaded from log, we return self.log_data["initial_drone_locations"].
            Otherwise, we call the parent's get_initial_drone_locations(),
            unify the format, and store it in the log.
            """

            # If we already have them in the log, just return it
            if self.loaded_from_log and self.log_data["initial_drone_locations"] is not None:
                return self.log_data["initial_drone_locations"]

            # otherwise, call the parent's method
            raw_locations = super().get_initial_drone_locations()

            # unify format to a list of (state, (x,y))
            init_list = self._normalize_initial_locations(raw_locations)

            # store in self.log_data
            self.log_data["initial_drone_locations"] = init_list

            # write to file
            self._save_log()

            # return as original style: if user's parent returns a 2-tuple, we do that. 
            # or if it returns a single list, we do that. 
            # but you have the parent call's raw format, so let's be consistent.
            # print(f"[wrap_log_drone_strategy] ✏️ Logging initial drone positions to {self.log_file}")
            # for i, (state, pos) in enumerate(init_list):
                # print(f"  Drone {i}: {state} at {pos}")
            return raw_locations

        def next_actions(self, automatic_step_parameters, custom_step_parameters):
            """
            If loaded from log, return the stored actions for this step_counter (if present).
            Otherwise, call parent's next_actions and store the result.
            """
            # if we have enough data in actions_history, we can just return
            if self.loaded_from_log and self.step_counter < len(self.log_data["actions_history"]):
                actions = self.log_data["actions_history"][self.step_counter]
                self.step_counter += 1
                return self._unpack_actions(actions)

            # otherwise, call parent
            # print(f"[wrap_log_drone_strategy] Calling parent's next_actions")
            # print(f"len log_data: {len(self.log_data['actions_history'])}")
            # print(f"step_counter: {self.step_counter}")
            # print(f"log name: {self.log_file}")
            actions = super().next_actions(automatic_step_parameters, custom_step_parameters)

            # store in log_data
            self.log_data["actions_history"].append(self._normalize_actions(actions))

            # increment step
            self.step_counter += 1

            # save log
            self._save_log()
            # if self.loaded_from_log and self.step_counter < len(self.log_data["actions_history"]):
            #     # print(f"[wrap_log_drone_strategy] 📂 Loading step {self.step_counter} actions from log")
            # else:
            #     # print(f"[wrap_log_drone_strategy] ✏️ Logging actions at step {self.step_counter} to {self.log_file}")
            #     for i, (typ, param) in enumerate(actions):
            #         # print(f"  Drone {i}: {typ} {param}")
            return actions

        ###############
        # HELPER FUNCS
        ###############

        def _save_log(self):
            with open(self.log_file, "w") as f:
                json.dump(self.log_data, f, indent=2)
            # print(f"[wrap_log_drone_strategy] 💾 Log updated and written to {self.log_file}")
            
        def _normalize_initial_locations(self, raw):
            """
            Convert raw output from parent's get_initial_drone_locations()
            into a standard list-of-lists format:
            e.g. [("charge",(x,y)), ("fly",(xx,yy)), ... ]
            or a single list if user returns that.

            parent might return:
               1) a single list => [("charge",(x,y)), ...] or just [(x,y), ...]
               2) a 2-tuple => ([ (x,y),...], [ state, ... ])

            We'll unify it so that in self.log_data, we store 
               a single list of (state,(x,y)) 
            """
            if isinstance(raw, list):
                # a single list
                # check if the first element is a 2-tuple (e.g. (x,y)) or (state,(x,y))
                if len(raw) == 0:
                    return []  # empty
                first = raw[0]
                if isinstance(first, tuple):
                    if len(first) == 2 and isinstance(first[0], str):
                        # e.g. ("charge",(x,y))
                        # we can store as is
                        return raw
                    elif len(first) == 2 and isinstance(first[0], (int, float)):
                        # e.g. (x,y)
                        # so let's store them as ("charge",(x,y)) by default
                        # print("set to be charge default for (x,y) tuples")
                        newlist = [("charge",(int(x),int(y))) for (x,y) in raw]
                        return newlist
                    else:
                        # fallback
                        return raw
                else:
                    # fallback
                    return raw

            elif isinstance(raw, tuple) and len(raw) == 2:
                # means (positions, states)
                positions, states = raw
                # build a single list of e.g. [(state,(x,y)), ...]
                combined = []
                for (x,y), st in zip(positions, states):
                    combined.append((st, (int(x),int(y))))
                return combined
            else:
                # fallback unknown
                return []

        def _unpack_initial_locations(self, stored):
            """
            Convert from our stored format (list of (state,(x,y))) 
            back to the parent's original style.

            In your DroneRoutingOptimizationModelReuseIndex, you typically do 
              return (positions, states).

            So let's do that.
            """
            # stored is a single list => [("charge",(x,y)), ...]

            # print(f"[wrap_log_drone_strategy] 📦 Loaded initial drone positions from log:")
            # for i, (st, (x, y)) in enumerate(stored):
                # print(f"  Drone {i}: {st} at ({x}, {y})")
            positions = []
            states = []
            for (st,(x,y)) in stored:
                positions.append((x,y))
                states.append(st)
            return positions, states

        def _normalize_actions(self, actions):
            """
            Convert a parent's return actions 
              e.g. [('move',(1,0)),('charge',(2,2))] 
            into a JSON-friendly format.

            We'll basically store them as the same structure:
              [("move",[1,0]), ("charge",[2,2])]
            but ensure coords are int or lists
            """
            out = []
            for (typ, param) in actions:
                if param is None:
                    out.append([typ, None])
                elif isinstance(param, tuple):
                    # e.g. (x,y)
                    out.append([typ, list(param)])
                else:
                    out.append([typ, param]) # fallback
            return out

        def _unpack_actions(self, stored):
            """
            Reverse of _normalize_actions:
             e.g.  [["move",[1,0]],["charge",[2,2]]] => [('move',(1,0)),('charge',(2,2))]
            """
            out = []
            for [typ, param] in stored:
                if param is None:
                    out.append((typ, None))
                else:
                    out.append((typ, tuple(param)))
            return out

    return LoggedDroneRoutingStrategy


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
        strategy_name = self.base_strategy_cls.__name__
        N = automatic_initialization_parameters.get("N", 0)
        M = automatic_initialization_parameters.get("M", 0)
        n_ground = automatic_initialization_parameters.get("n_ground_stations", 0)
        n_charging = automatic_initialization_parameters.get("n_charging_stations", 0)

        log_filename = f"{layout_name}_{strategy_name}_{N}N_{M}M_{n_ground}ground_{n_charging}charge.json"
        log_path = os.path.join(log_dir, log_filename)
        self.log_path = log_path

        self.ground_sensor_locations = []
        self.charging_station_locations = []

        if os.path.exists(log_path):
            print(f"[LoggableSensorStrategy] Loaded placement from: {log_path}")
            with open(log_path, "r") as f:
                data = json.load(f)
                self.ground_sensor_locations = [tuple(loc) for loc in data["ground_sensor_locations"]]
                self.charging_station_locations = [tuple(loc) for loc in data["charging_station_locations"]]
        else:
            print(f"[LoggableSensorStrategy] Running strategy: {strategy_name}")
            base = self.base_strategy_cls(automatic_initialization_parameters, custom_initialization_parameters)
            self.ground_sensor_locations, self.charging_station_locations = base.get_locations()
            with open(log_path, "w") as f:
                json.dump({
                    "ground_sensor_locations": self.ground_sensor_locations,
                    "charging_station_locations": self.charging_station_locations
                }, f, indent=2)
            print(f"[LoggableSensorStrategy] Saved placements to: {log_path}")

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
            '__module__'      : __name__,             # <-- crucial
        }
    )

    # Register the class in the *module* namespace so that pickle
    # can import it from "wrappers.Wrapped" later.
    globals()[name] = Wrapped

    return Wrapped


def _deep_unwrap(cls):
    """
    Recursively unwrap any wrapper class until we reach the base strategy.
    Works for custom wrappers like get_wrapped_clustering_strategy.
    """
    seen = set()
    while True:
        if hasattr(cls, "base_strategy_cls"):
            cls = cls.base_strategy_cls
        elif hasattr(cls, "__wrapped__"):  # optional: for functools.wraps
            cls = cls.__wrapped__
        elif hasattr(cls, "__origin__"):  # optional: for generic wrappers
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
        N, M          = automatic_initialization_parameters["N"], automatic_initialization_parameters["M"]
        n_drones      = automatic_initialization_parameters["n_drones"]
        n_charge      = automatic_initialization_parameters["n_charging_stations"]
        n_ground      = automatic_initialization_parameters["n_ground_stations"]

        # fingerprint of charging-station layout (sorted, Julia → Python index already)
        layout_fp     = "_".join([f"{x}-{y}" for x, y
                                  in sorted(automatic_initialization_parameters["charging_stations_locations"])])

        self._log_path = os.path.join(
            log_dir,
            f"{os.path.basename(burnmap_file)}_{strategy_name}_{n_drones}_drones_"
            f"{n_charge}_charging_stations_{n_ground}_ground_stations_{layout_fp}_"
            "logged_drone_routing.json"
        )

        # -----------------------------------------------------------------
        # load or create log structure
        # -----------------------------------------------------------------
        self._step_counter  = 0
        self._loaded_from_disk = False

        if (not custom_initialization_parameters.get("recompute_logfile", False)
                and os.path.exists(self._log_path)):
            with open(self._log_path, "r") as fp:
                self._log = json.load(fp)
            self._loaded_from_disk = True
            # sanity: convert lists→tuples where handy
            if self._log.get("initial_drone_locations"):
                self._log["initial_drone_locations"] = [
                    (st, tuple(pos)) for st, pos in self._log["initial_drone_locations"]
                ]
        else:
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
                return [(st, (int(x), int(y))) for st, (x, y) in raw]
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
    name = f"{strategy_cls.__name__}Logged"

    Wrapped = type(
        name,
        (LoggableDroneStrategyWrapper,),
        {
            'base_strategy_cls': strategy_cls,
            'strategy_name'   : strategy_cls.__name__,
            '__module__'      : __name__,
        }
    )

    globals()[name] = Wrapped
    return Wrapped


from Strategy import RandomSensorPlacementStrategy, DroneRoutingUniformCoverageResetStatic
from new_clustering import get_wrapped_clustering_strategy

# Register statically at module load

RandomSensorPlacementStrategyLogged = make_loggable_sensor_strategy(RandomSensorPlacementStrategy)
DroneRoutingUniformCoverageResetStaticLogged = make_loggable_drone_strategy(get_wrapped_clustering_strategy(DroneRoutingUniformCoverageResetStatic))


# ====================
# Sensor Wrapper
# ====================

# class LoggableSensorStrategyWrapper:
#     def __init__(self, strategy_cls, auto_params: dict, custom_params: dict):
#         self.strategy_cls = strategy_cls
#         self.auto_params = auto_params
#         self.custom_params = custom_params
#         self._initialize()

#     def _initialize(self):
#         burnmap_path = self.custom_params.get("burnmap_filename")
#         if not burnmap_path:
#             raise ValueError("Expected 'burnmap_filename' in custom_initialization_parameters")

#         base_dir = os.path.dirname(os.path.abspath(burnmap_path))
#         log_dir = os.path.join(base_dir, "logs")
#         os.makedirs(log_dir, exist_ok=True)

#         layout_name = self.custom_params.get("log_file", "layout")
#         strategy_name = self.strategy_cls.__name__
#         N = self.auto_params.get("N", 0)
#         M = self.auto_params.get("M", 0)
#         n_ground = self.auto_params.get("n_ground_stations", 0)
#         n_charging = self.auto_params.get("n_charging_stations", 0)

#         log_filename = f"{layout_name}_{strategy_name}_{N}N_{M}M_{n_ground}ground_{n_charging}charge.json"
#         log_path = os.path.join(log_dir, log_filename)
#         self.log_path = log_path

#         if os.path.exists(log_path):
#             print(f"[Sensor] Loaded placement from: {log_path}")
#             with open(log_path, "r") as f:
#                 data = json.load(f)
#                 self.ground_sensor_locations = [tuple(loc) for loc in data["ground_sensor_locations"]]
#                 self.charging_station_locations = [tuple(loc) for loc in data["charging_station_locations"]]
#         else:
#             print(f"[Sensor] Running strategy: {strategy_name}")
#             strategy_instance = self.strategy_cls(self.auto_params, self.custom_params)
#             self.ground_sensor_locations, self.charging_station_locations = strategy_instance.get_locations()
#             with open(log_path, "w") as f:
#                 json.dump({
#                     "ground_sensor_locations": self.ground_sensor_locations,
#                     "charging_station_locations": self.charging_station_locations
#                 }, f, indent=2)
#             print(f"[Sensor] Saved placements to: {log_path}")

#     def get_locations(self):
#         return self.ground_sensor_locations, self.charging_station_locations


# # ====================
# # Drone Wrapper
# # ====================

# class LoggableDroneStrategyWrapper:
#     def __init__(self, strategy_cls, auto_params: dict, custom_params: dict):
#         self._inner = strategy_cls(auto_params, custom_params)
#         self.custom_params = custom_params
#         self.auto_params = auto_params
#         self._step_counter = 0
#         self._loaded_from_disk = False
#         self._init_log()

#     def _init_log(self):
#         burnmap_file = self.custom_params["burnmap_filename"]
#         base_dir = os.path.dirname(os.path.abspath(burnmap_file))
#         log_dir = os.path.join(base_dir, "logs")
#         os.makedirs(log_dir, exist_ok=True)

#         strategy_name = self._inner.__class__.__name__
#         N = self.auto_params["N"]
#         M = self.auto_params["M"]
#         n_drones = self.auto_params["n_drones"]
#         n_charge = self.auto_params["n_charging_stations"]
#         n_ground = self.auto_params["n_ground_stations"]
#         layout_fp = "_".join([f"{x}-{y}" for x, y in sorted(self.auto_params["charging_stations_locations"])])

#         self._log_path = os.path.join(
#             log_dir,
#             f"{os.path.basename(burnmap_file)}_{strategy_name}_{n_drones}_drones_"
#             f"{n_charge}_charging_stations_{n_ground}_ground_stations_{layout_fp}_logged_drone_routing.json"
#         )

#         if not self.custom_params.get("recompute_logfile", False) and os.path.exists(self._log_path):
#             with open(self._log_path, "r") as fp:
#                 self._log = json.load(fp)
#             self._loaded_from_disk = True
#             if self._log.get("initial_drone_locations"):
#                 self._log["initial_drone_locations"] = [
#                     (st, tuple(pos)) for st, pos in self._log["initial_drone_locations"]
#                 ]
#         else:
#             self._log = {
#                 "initial_drone_locations": None,
#                 "actions_history": []
#             }

#     def get_initial_drone_locations(self):
#         if self._log["initial_drone_locations"] is not None:
#             return self._log["initial_drone_locations"]

#         raw = self._inner.get_initial_drone_locations()
#         self._log["initial_drone_locations"] = self._normalize_initial(raw)
#         self._flush()
#         return raw

#     def next_actions(self, auto_step_params, custom_step_params):
#         if self._loaded_from_disk and self._step_counter < len(self._log["actions_history"]):
#             stored = self._log["actions_history"][self._step_counter]
#             self._step_counter += 1
#             return self._denormalize_actions(stored)

#         acts = self._inner.next_actions(auto_step_params, custom_step_params)
#         self._log["actions_history"].append(self._normalize_actions(acts))
#         self._step_counter += 1
#         self._flush()
#         return acts

#     def __getattr__(self, item):
#         return getattr(self._inner, item)

#     def _flush(self):
#         with open(self._log_path, "w") as fp:
#             json.dump(self._log, fp, indent=2)

#     @staticmethod
#     def _normalize_initial(raw) -> List[Tuple[str, Tuple[int, int]]]:
#         if isinstance(raw, list):
#             if raw and isinstance(raw[0], tuple) and isinstance(raw[0][0], str):
#                 return [(st, (int(x), int(y))) for st, (x, y) in raw]
#             return [("charge", (int(x), int(y))) for (x, y) in raw]
#         if isinstance(raw, tuple) and len(raw) == 2:
#             positions, states = raw
#             return [(st, (int(x), int(y))) for (x, y), st in zip(positions, states)]
#         raise ValueError("Unexpected initial-location format from strategy")

#     @staticmethod
#     def _normalize_actions(acts):
#         return [[typ, None if param is None else list(param)] for typ, param in acts]

#     @staticmethod
#     def _denormalize_actions(stored):
#         return [(typ, None if param is None else tuple(param)) for typ, param in stored]


# # ====================
# # Top-Level Factories
# # ====================

# class SensorStrategyFactory:
#     def __init__(self, strategy_cls):
#         self.strategy_cls = strategy_cls

#     def __call__(self, auto_params, custom_params):
#         return LoggableSensorStrategyWrapper(self.strategy_cls, auto_params, custom_params)


# class DroneStrategyFactory:
#     def __init__(self, strategy_cls):
#         self.strategy_cls = strategy_cls

#     def __call__(self, auto_params, custom_params):
#         return LoggableDroneStrategyWrapper(self.strategy_cls, auto_params, custom_params)


# def create_sensor_strategy(strategy_cls):
#     return SensorStrategyFactory(strategy_cls)


# def create_drone_strategy(strategy_cls):
#     return DroneStrategyFactory(strategy_cls)