import math
import os
import json
import random
import matplotlib.pyplot as plt
from collections import deque

# Shapely for merging bounding boxes
from shapely.geometry import Polygon
from shapely.ops import unary_union

##############################################################################
# 1) BFS to find clusters
##############################################################################
def find_clusters(charging_stations, drone_battery):
    """
    BFS-based method to group charging stations.
    Two stations are in the same cluster if distance <= (drone_battery / 2).
    
    Returns: 
      A list of clusters, each is a list of station coords [(x,y),...].
    """
    n = len(charging_stations)
    radius = drone_battery / 2.0
    adjacency_list = [[] for _ in range(n)]

    for i in range(n):
        x1, y1 = charging_stations[i]
        for j in range(i+1, n):
            x2, y2 = charging_stations[j]
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist <= radius:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    cluster_labels = [-1]*n
    current_cluster = 0
    for start_node in range(n):
        if cluster_labels[start_node] == -1:
            queue = deque([start_node])
            cluster_labels[start_node] = current_cluster
            while queue:
                node = queue.popleft()
                for neighbor in adjacency_list[node]:
                    if cluster_labels[neighbor] == -1:
                        cluster_labels[neighbor] = current_cluster
                        queue.append(neighbor)
            current_cluster += 1

    # group stations by cluster label
    clusters = [[] for _ in range(current_cluster)]
    for i, lbl in enumerate(cluster_labels):
        clusters[lbl].append(charging_stations[i])

    return clusters

##############################################################################
# 2) Use shapely to build union bounding boxes for each cluster
##############################################################################
def get_cluster_boundary_boxes(stations, half_extent):
    """
    For each station in 'stations', build a bounding box of side 2*half_extent.
    Union them -> list of Polygon(s).
    """
    box_polygons = []
    for (x, y) in stations:
        rect = Polygon([
            (x - half_extent, y - half_extent),
            (x + half_extent, y - half_extent),
            (x + half_extent, y + half_extent),
            (x - half_extent, y + half_extent),
        ])
        box_polygons.append(rect)

    union_poly = unary_union(box_polygons)
    if union_poly.is_empty:
        return []
    if union_poly.geom_type == "Polygon":
        return [union_poly]
    else:
        return list(union_poly)

##############################################################################
# Utility: from union polygon(s) => get integer grid size (N, M)
##############################################################################
def get_bounding_grid_size(polygons):
    """
    polygons: list of shapely Polygon objects (already unioned if you like).
    We take the union and then read .bounds => (minx, miny, maxx, maxy).
    Then we define:
      N = (maxx - minx + 1), M = (maxy - miny + 1), 
    after rounding outward so we have integer coverage.
    
    Returns: (N, M, min_x_int, min_y_int)
       Where you can do further logic if you want to shift station coords.
    """
    if not polygons:
        # fallback
        return (1, 1, 0, 0)

    union_poly = unary_union(polygons)
    minx, miny, maxx, maxy = union_poly.bounds
    
    # round them outward
    min_x_int = math.floor(minx)
    min_y_int = math.floor(miny)
    max_x_int = math.ceil(maxx)
    max_y_int = math.ceil(maxy)

    N = max_x_int - min_x_int + 1
    M = max_y_int - min_y_int + 1
    
    return (N, M, min_x_int, min_y_int)


from shapely.geometry import Point
import random

def sample_points_in_polygon(poly, n_points, max_tries=10_000):
    """
    Sample n_points uniformly at random in 'poly' by bounding-box rejection sampling.
    poly: shapely Polygon or MultiPolygon
    n_points: how many points you want
    max_tries: to prevent infinite loops if the polygon is too small

    Returns a list of (x, y) floating coords.
    """
    # Merge if it's multi
    unioned = poly
    if unioned.geom_type == "MultiPolygon":
        unioned = unioned.union()  # or unary_union

    minx, miny, maxx, maxy = unioned.bounds

    samples = []
    tries = 0
    while len(samples) < n_points and tries < max_tries:
        tries += 1
        # pick random x,y in bounding box
        rx = random.uniform(minx, maxx)
        ry = random.uniform(miny, maxy)
        candidate = Point(rx, ry)

        # check if inside polygon
        if candidate.within(unioned):
            samples.append((rx, ry))

    return samples

##############################################################################
# 3) Minimal random sensor strategy for demonstration
##############################################################################
class RandomSensorPlacementStrategy:
    """
    Sensor placement strategy that:
      - Places ground sensors randomly.
      - Uses the charging stations from automatic_init_params["charging_stations_locations"].
    """
    def __init__(self, automatic_initialization_parameters: dict, custom_initialization_parameters: dict):
        """
        Args:
            automatic_initialization_parameters: dict with keys:
                "N": Grid height
                "M": Grid width
                "min_x": the lower left corner of that clusters bounding box in x global coordinate
                "min_y": the lower left corner of that clusters bounding box in y global coordinate
                "n_ground_stations": number of ground sensors to place
                "charging_stations_locations": List of (x,y) for charging stations
                ... (other fields as needed)
            custom_initialization_parameters: dict
        """
        # Save references for convenience
        self.N = automatic_initialization_parameters["N"]
        self.M = automatic_initialization_parameters["M"]
        self.cluster_polygon = automatic_initialization_parameters["cluster_polygon"]  # a shapely Polygon or MultiPolygon
        self.n_ground_stations = automatic_initialization_parameters["n_ground_stations"]

        # 1) Randomly place ground sensors
        self.ground_sensor_locations = sample_points_in_polygon(
            self.cluster_polygon, 
            self.n_ground_stations
        )

        # 2) Use the existing station locations from the auto init param
        #    (instead of generating them randomly).
        self.charging_station_locations = automatic_initialization_parameters["charging_stations_locations"]

    def get_locations(self):
        """
        Returns the locations of the ground sensors and charging stations.
        """
        return self.ground_sensor_locations, self.charging_station_locations

##############################################################################
# 4) Your LoggedDroneRoutingStrategy (unchanged code)
##############################################################################
class LoggedDroneRoutingStrategy:
    """
    LoggedDroneRoutingStrategy logs drone routing actions and locations at every timestep.
    """
    def __init__(self, automatic_initialization_parameters, custom_initialization_parameters):
        self.automatic_initialization_parameters = automatic_initialization_parameters
        self.custom_initialization_parameters = custom_initialization_parameters

        if "burnmap_filename" not in custom_initialization_parameters:
            raise ValueError("Missing 'burnmap_filename' in custom_initialization_parameters")
        if "call_every_n_steps" not in custom_initialization_parameters:
            raise ValueError("Missing 'call_every_n_steps' in custom_initialization_parameters")
        if "optimization_horizon" not in custom_initialization_parameters:
            raise ValueError("Missing 'optimization_horizon' in custom_initialization_parameters")

        self.call_every_n_steps = custom_initialization_parameters["call_every_n_steps"]
        self.optimization_horizon = custom_initialization_parameters["optimization_horizon"]

        self.call_counter = 0
        self.current_solution = []  

        # LOG FILE SETUP
        if "log_file" in custom_initialization_parameters:
            log_file_path = custom_initialization_parameters["log_file"]
            log_dir = os.path.dirname(log_file_path)
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = log_file_path
        else:
            N = self.automatic_initialization_parameters.get("N", "N")
            M = self.automatic_initialization_parameters.get("M", "M")
            n_drones = self.automatic_initialization_parameters.get("n_drones", 0)
            n_charging_stations = self.automatic_initialization_parameters.get("n_charging_stations", 0)

            log_filename = f"drone_strategy_{N}N_{M}M_{n_drones}drones_{n_charging_stations}charge.json"
            log_dir = custom_initialization_parameters.get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, log_filename)

        self.log_data = {
            "initial_drone_locations": None,
            "steps": []
        }
        print(f"[LoggedDroneRoutingStrategy] Initialized with log file: {self.log_file}")

    def get_initial_drone_locations(self):
        charging_stations = self.automatic_initialization_parameters["charging_stations_locations"]
        n_drones = self.automatic_initialization_parameters["n_drones"]

        if len(charging_stations) == 0:
            initial_positions = [(0,0)] * n_drones
        else:
            n_stations = len(charging_stations)
            q = n_drones // n_stations
            r = n_drones % n_stations
            initial_positions = charging_stations * q + charging_stations[:r]

        self.log_data["initial_drone_locations"] = initial_positions
        self._write_log_to_file()
        return initial_positions

    def next_actions(self, automatic_step_parameters, custom_step_parameters):
        if self.call_counter % self.call_every_n_steps == 0:
            _, self.current_solution = self.dummy_drone_routing_robust(
                automatic_step_parameters, custom_step_parameters
            )

        timestep_index = self.call_counter % self.call_every_n_steps
        actions = self.current_solution[timestep_index]

        self._log_timestep(
            timestep=automatic_step_parameters["t"],
            drone_locations=automatic_step_parameters["drone_locations"],
            drone_batteries=automatic_step_parameters["drone_batteries"],
            actions=actions
        )

        self.call_counter += 1
        return actions

    def dummy_drone_routing_robust(self, automatic_step_parameters, custom_step_parameters):
        print("[Dummy Function] Generating dummy routing solution...")

        n_drones = self.automatic_initialization_parameters.get("n_drones", 3)
        n_timesteps = self.optimization_horizon

        # We'll produce an action set for each of the n_timesteps
        actions_per_timestep = []
        for t in range(n_timesteps):
            step_actions = []
            for d in range(n_drones):
                if t % 2 == 0:
                    step_actions.append(('move', (1, 0)))
                else:
                    step_actions.append(('charge', None))
            actions_per_timestep.append(step_actions)

        return [], actions_per_timestep

    def _log_timestep(self, timestep, drone_locations, drone_batteries, actions):
        log_entry = {
            "timestep": timestep,
            "drone_locations": drone_locations,
            "drone_batteries": drone_batteries,
            "actions": actions
        }
        self.log_data["steps"].append(log_entry)
        self._write_log_to_file()

    def _write_log_to_file(self):
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=2)

##############################################################################
# 5) The main function that:
#    - BFS to find clusters
#    - For each cluster, picks # drones from your 'drones_per_cluster' list
#    - Instantiates LoggedDroneRoutingStrategy and a sensor strategy
#    - Runs next_actions for a few timesteps
#    - Builds bounding polygons
#    - Plots everything in one final figure
##############################################################################
def run_and_plot(
    charging_stations, 
    drone_battery, 
    drones_per_cluster,
    timesteps=5,
    half_extent=2.0
):
    """
    charging_stations : list of (x,y)
    drone_battery     : float
    drones_per_cluster: list of ints, # of drones in each cluster
    timesteps         : how many timesteps to run .next_actions
    half_extent       : bounding box expansion for polygons

    We'll BFS -> get clusters -> check that len(clusters) == len(drones_per_cluster).
    Then run sensor + logged drone strategy on each cluster in turn.
    We'll store final states for plotting.
    """

    clusters = find_clusters(charging_stations, drone_battery)
    num_clusters = len(clusters)
    if num_clusters != len(drones_per_cluster):
        raise ValueError(
            f"Mismatch: Found {num_clusters} clusters but drones_per_cluster has length {len(drones_per_cluster)}"
        )

    print(f"Found {num_clusters} cluster(s). Running strategies...")

    # We'll store info for plotting
    cluster_data = []
    color_map = ["blue", "orange", "green", "red", "purple", "cyan", "magenta", "gray"]

    # We'll do 3 timesteps for demonstration
    timesteps = 3

    for cid, stations_in_cluster in enumerate(clusters):

        color = color_map[cid % len(color_map)]
        print(f"\n=== Cluster {cid} => stations: {stations_in_cluster}")
        n_drones_for_this_cluster = drones_per_cluster[cid]
        # 1) build bounding polygons for that cluster
        polygons = get_cluster_boundary_boxes(stations_in_cluster, half_extent)
        # unify them if multiple
        cluster_polygon = unary_union(polygons)
        # 2) from those polygons => get integer N,M
        N, M, min_x, min_y = get_bounding_grid_size(polygons)
        print(f"   bounding box => N={N}, M={M}, (min_x={min_x}, min_y={min_y})")


       # 3) Build auto init
        auto_init_params = {
            "N": N,
            "M": M,
            "min_x": min_x,
            "min_y": min_y,
            "max_battery_distance": 100,
            "max_battery_time": 100,
            "n_drones": n_drones_for_this_cluster,
            "n_ground_stations": 1,
            "n_charging_stations": len(stations_in_cluster),
            "ground_sensor_locations": [],
            "cluster_polygon": cluster_polygon,
            # we keep station coords as-is, but they might lie outside [0..N-1].
            # for a full solution, you might SHIFT them so min_x->0, etc.
            "charging_stations_locations": stations_in_cluster,
        }

        custom_init_params = {
            "burnmap_filename": "dummy.txt",
            "call_every_n_steps": 1,
            "optimization_horizon": timesteps,
            # we skip log_file for simplicity
        }

        # Create sensor strategy & retrieve sensor placements
        sensor_strat = RandomSensorPlacementStrategy(auto_init_params, custom_init_params)
        ground_sensors, placed_charging_stations = sensor_strat.get_locations()

        # Create the LoggedDroneRoutingStrategy
        routing_strat = LoggedDroneRoutingStrategy(auto_init_params, custom_init_params)

        # 1) get drone initial
        initial_drone_positions = routing_strat.get_initial_drone_locations()
        drone_locs = list(initial_drone_positions)
        drone_batts = [(100,100)] * n_drones_for_this_cluster  # dummy battery
        # 2) run next_actions for a few timesteps
        for t in range(timesteps):
            auto_step_params = {
                "drone_locations": drone_locs,
                "drone_batteries": drone_batts,
                "t": t
            }
            custom_step_params = {}
            actions = routing_strat.next_actions(auto_step_params, custom_step_params)
            # apply moves
            for i, act in enumerate(actions):
                if act[0] == 'move':
                    dx, dy = act[1]
                    oldx, oldy = drone_locs[i]
                    newx, newy = oldx + dx, oldy + dy
                    # no boundary clamp for brevity
                    drone_locs[i] = (newx, newy)
                elif act[0] == 'charge':
                    # pretend we recharge
                    drone_batts[i] = (100,100)

        # store everything for plotting
        cluster_data.append({
            "cid": cid,
            "color": color,
            "stations": stations_in_cluster,
            "ground_sensors": ground_sensors,
            "charging_stations": placed_charging_stations,
            "init_drone_positions": initial_drone_positions,
            "final_drone_positions": drone_locs,
            "polygons": polygons,
        })

    # --------------------------------------------------
    # Plot everything
    # --------------------------------------------------
    plt.figure(figsize=(8,8))
    plt.title(f"BFS Clusters, #drones per cluster, T={timesteps} steps")

    print("cluster_data", cluster_data)

    for cinfo in cluster_data:
        cid = cinfo["cid"]
        color = cinfo["color"]

        # stations
        sx = [p[0] for p in cinfo["stations"]]
        sy = [p[1] for p in cinfo["stations"]]
        plt.scatter(sx, sy, c=color, marker='o', s=80, label=f"Cluster {cid} stations" if cid==0 else None)

        # ground sensors
        gx = [p[0] for p in cinfo["ground_sensors"]]
        gy = [p[1] for p in cinfo["ground_sensors"]]
        plt.scatter(gx, gy, c=color, marker='s', s=120, edgecolors='black', 
                    label=f"Cluster {cid} ground sensors" if cid==0 else None)

        # placed charging
        cx = [p[0] for p in cinfo["charging_stations"]]
        cy = [p[1] for p in cinfo["charging_stations"]]
        plt.scatter(cx, cy, c=color, marker='*', s=140, edgecolors='black',
                    label=f"Cluster {cid} charging stn" if cid==0 else None)

        # initial drone loc
        idx = [p[0] for p in cinfo["init_drone_positions"]]
        idy = [p[1] for p in cinfo["init_drone_positions"]]
        plt.scatter(idx, idy, c=color, marker='D', s=80, edgecolors='black',
                    label=f"Cluster {cid} initial drones" if cid==0 else None)

        # final drone loc
        fdx = [p[0] for p in cinfo["final_drone_positions"]]
        fdy = [p[1] for p in cinfo["final_drone_positions"]]
        plt.scatter(fdx, fdy, c=color, marker='D', s=80, edgecolors='red',
                    label=f"Cluster {cid} final drones" if cid==0 else None)

        # bounding polygons
        for poly in cinfo["polygons"]:
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.2, facecolor=color, edgecolor='black')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', 'box')
    plt.show()


##############################################################################
# Example usage
##############################################################################
if __name__ == "__main__":
    # Example stations
    stations = [
        (2,2), (3,2), (4,4),
        (10,10), (11,10),
        (22,22)
    ]
    drone_battery = 6.0

    # Suppose we have 3 clusters from BFS, so we specify # drones for each cluster:
    # e.g. cluster 0 => 2 drones, cluster 1 => 1 drone, cluster 2 => 3 drones
    drones_per_cluster = [2, 1, 3]

    # We'll run 5 timesteps, bounding boxes with half_extent=2.0 
    run_and_plot(stations, drone_battery, drones_per_cluster, timesteps=5, half_extent=2.0)
