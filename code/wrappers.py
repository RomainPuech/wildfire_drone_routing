import importlib.util
import os
import tqdm
import json

### For SensorPlacement Strategies
def wrap_log_sensor_strategy(input_strat_cls):
    """
    Wraps a SensorPlacementStrategy to log and reuse previous placements.

    Args:
        input_strat_cls (SensorPlacementStrategy): The input sensor placement strategy class.

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
                custom_initialization_parameters: dict
                    Expected keys:
                        - log_file: Path to the log file
                        - burnmap_filename: Path to the burn map used by the Julia optimizer
            """

            n_ground = automatic_initialization_parameters.get("n_ground_stations", 0)
            n_charging = automatic_initialization_parameters.get("n_charging_stations", 0)
            N = automatic_initialization_parameters.get("N", 0)
            M = automatic_initialization_parameters.get("M", 0)
            strategy_name = input_strat_cls.__name__

            # Save logs next to burnmap in "logs" directory
            log_dir = os.path.join(os.path.dirname(custom_initialization_parameters["burnmap_filename"]), "logs")
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, f"{strategy_name}_{N}N_{M}M_{n_ground}ground_{n_charging}charge.json")

            
            self.ground_sensor_locations = []
            self.charging_station_locations = []

            if os.path.exists(log_path):
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


def wrap_log_drone_strategy(input_drone_cls):
    """
    Wraps a DroneRoutingStrategy so that:
      - get_initial_drone_locations() and next_actions() are logged to a JSON file
      - if the file exists, we load from it (no re-optimization needed)
    
    Specifically compatible with strategies that:
      - either return a single list of (state,(x,y)) from get_initial_drone_locations()
      - or return a 2-tuple (positions, states), e.g. ([ (x1,y1), ...], ["charge","fly",...])
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
            log_dir = os.path.join(os.path.dirname(custom_initialization_parameters["burnmap_filename"]), "logs")
            os.makedirs(log_dir, exist_ok=True)

            # Create a fingerprint string based on charging station layout
            charging_stations = automatic_initialization_parameters["charging_stations_locations"]
            layout_fingerprint = "_".join([f"{x}-{y}" for x, y in sorted(charging_stations)])

            # Build full filename
            log_name = f"{input_drone_cls.strategy_name}_" + \
                    f"{automatic_initialization_parameters['n_drones']}_drones_" + \
                    f"{automatic_initialization_parameters['n_charging_stations']}_charging_stations_" + \
                    f"{automatic_initialization_parameters['n_ground_stations']}_ground_stations_" + \
                    layout_fingerprint + "_" + \
                    (f"{custom_initialization_parameters['optimization_horizon']}_" if 'optimization_horizon' in custom_initialization_parameters else '') + "_" + \
                    (f"{custom_initialization_parameters['reevaluation_step']}_" if 'reevaluation_step' in custom_initialization_parameters else '') + \
                    (f"{custom_initialization_parameters['regularization_param']}_" if 'regularization_param' in custom_initialization_parameters else 'no_regularization') + \
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

            # return as original style: if user’s parent returns a 2-tuple, we do that. 
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
