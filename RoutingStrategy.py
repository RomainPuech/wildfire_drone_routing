import random
# from my_julia_caller import jl # DEACTIVATED TO RUN THINGS IN PARALLEL 
import json


class RoutingStrategy():
    # Routing_Strat.next_actions(drone_locations,drone_battery,t_found)

    def __init__(self,ground_sensor_locations,charging_stations_locations,n_drones=None, max_battery_distance = 100, max_battery_time = 100):
          self.ground_sensor_locations = ground_sensor_locations
          self.charging_stations_locations = charging_stations_locations
          self.n_drones = n_drones if not n_drones is None else len(charging_stations_locations)
          self.drone_locations = []
          self.drone_batteries = [(max_battery_distance,max_battery_time) for i in range(n_drones)]

    def initialize(self):
        n = len(self.charging_stations_locations)
        q = self.n_drones // n
        r = self.n_drones % n
        self.drone_locations = self.charging_stations_locations*q + self.charging_stations_locations[:r]
        return self.drone_locations
    
    def next_actions(self, drone_locations = None, drone_batteries = None, t=0):
        if drone_locations is None:
             drone_locations = self.drone_locations
        if drone_batteries is None:
             drone_batteries = self.drone_batteries

        # suggest actions
        return [('move',(random.randint(-5,5),random.randint(-5,5))) for _ in range(self.n_drones)]
    

class GroundPlacementStrategy():
    def __init__(self, n_ground_stations, n_charging_stations, N):
        """
        Initialize the ground placement strategy using random placement.
        
        Args:
            n_ground_stations: Target number of ground stations
            n_charging_stations: Target number of charging stations
            N: Grid size
        """
        self.ground_sensor_locations = []
        self.charging_station_locations = []
        
        # Generate random positions for ground sensors
        for _ in range(n_ground_stations):
            x = random.randint(0, N-1)
            y = random.randint(0, N-1)
            self.ground_sensor_locations.append((x, y))
        
        # Generate random positions for charging stations
        for _ in range(n_charging_stations):
            x = random.randint(0, N-1)
            y = random.randint(0, N-1)
            self.charging_station_locations.append((x, y))

    def get_locations(self):
        return self.ground_sensor_locations, self.charging_station_locations

class GroundPlacementOptimization(GroundPlacementStrategy):
    def __init__(self, n_ground_stations, n_charging_stations, N, risk_pertime_dir):
        """
        Initialize the ground placement strategy using Julia's optimization model.
        
        Args:
            n_ground_stations: Target number of ground stations
            n_charging_stations: Target number of charging stations
            N: Grid size
            risk_pertime_dir: Directory containing burn map files
        """
        # Initialize empty lists (skip parent's random initialization)
        self.ground_sensor_locations = []
        self.charging_station_locations = []
        
        # Load the Julia module and function
     #    jl.include("julia/ground_charging_opt.jl") # Done in julia_caller
        
        # Call the Julia optimization function
        x_vars, y_vars = jl.ground_charging_opt_model_grid(risk_pertime_dir, n_ground_stations, n_charging_stations)
        self.ground_sensor_locations = list(x_vars)
        self.charging_station_locations = list(y_vars)

lognb = 0
class LoggedGroundPlacementStrategy(GroundPlacementStrategy):
        def __init__(self, logfile):
            global lognb
            with open(logfile) as log:
                 self.ground_sensor_locations, self.charging_station_locations = json.load(log)[lognb]
            lognb +=1
           
          
def reset_count():
    global lognb
    lognb = 0