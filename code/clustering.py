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
# Drone Routing Strategy Wrapper Class
##############################################################################
class DroneRoutingStrategyClusterWrapper:
    """
    A wrapper class that adds clustering functionality to any drone routing strategy.
    This wrapper provides methods for clustering, boundary box calculation, and visualization.
    """
    def __init__(self, StrategyClass):
        """
        Args:
            StrategyClass: The drone routing strategy class to wrap
        """
        self.StrategyClass = StrategyClass
        self.clusters = None
        self.cluster_data = None
        
    def find_clusters(self, charging_stations, drone_battery):
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

        self.clusters = clusters
        return clusters
        
    def get_cluster_boundary_boxes(self, stations, half_extent):
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
            
    def get_bounding_grid_size(self, polygons):
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
        
    def run_clusters(self, 
                 charging_stations, 
                 drone_battery, 
                 drones_per_cluster,
                 timesteps,
                 half_extent,
                 total_ground_sensors,
                 SensorStrategyClass,
                 custom_init_params=None):
        """
        Run the routing strategies per cluster, but do NOT plot.
        Just stores self.cluster_data so you can plot later.
        """
        if custom_init_params is None:
            custom_init_params = {
                "burnmap_filename": "dummy.txt",
                "call_every_n_steps": 1,
                "optimization_horizon": timesteps,
            }

        if self.clusters is None:
            self.clusters = self.find_clusters(charging_stations, drone_battery)

        # global bounding box
        all_polygons = []
        for stations_in_cluster in self.clusters:
            all_polygons += self.get_cluster_boundary_boxes(stations_in_cluster, half_extent)

        N, M, min_x, min_y = self.get_bounding_grid_size(all_polygons)

        auto_init_params_global = {
            "N": N,
            "M": M,
            "min_x": min_x,
            "min_y": min_y,
            "max_battery_distance": 100,
            "max_battery_time": 100,
            "n_ground_stations": total_ground_sensors,
            "n_charging_stations": len(charging_stations),
            "charging_stations_locations": charging_stations
        }

        sensor_strat = SensorStrategyClass(auto_init_params_global, custom_init_params)
        ground_sensors, _ = sensor_strat.get_locations()

        color_map = ["blue", "orange", "green", "red", "purple", "cyan", "magenta", "gray"]
        cluster_data = []
        strategy_instances = []

        for cid, stations_in_cluster in enumerate(self.clusters):
            print(f"\n🚀 Running cluster {cid} with {len(stations_in_cluster)} charging stations and {drones_per_cluster[cid]} drones")
            polygons = self.get_cluster_boundary_boxes(stations_in_cluster, half_extent)
            cluster_polygon = unary_union(polygons)

            N, M, min_x, min_y = self.get_bounding_grid_size(polygons)
            print(f"  🧱 Bounding grid: {N} x {M}, origin: ({min_x}, {min_y})")

            auto_params = {
                "N": N,
                "M": M,
                "min_x": min_x,
                "min_y": min_y,
                "max_battery_distance": 100,
                "max_battery_time": 100,
                "n_drones": drones_per_cluster[cid],
                "n_ground_stations": 1,
                "n_charging_stations": len(stations_in_cluster),
                "ground_sensor_locations": ground_sensors,
                "cluster_polygon": cluster_polygon,
                "charging_stations_locations": stations_in_cluster,
            }

            strat = self.StrategyClass(auto_params, custom_init_params)
            strategy_instances.append(strat)
            print(f"  ✅ Strategy initialized for cluster {cid}")
            # only call once
            drone_init_output = strat.get_initial_drone_locations()
            drone_locs = list(drone_init_output[0])
            drone_states = list(drone_init_output[1])
            drone_batts = [(100, 100)] * drones_per_cluster[cid]

            print(f"  🚁 Initial drone positions: {drone_init_output}")
            print(f"  🔋 Initial drone batteries: {drone_batts}")

            for t in range(timesteps):
                actions = strat.next_actions(
                    {
                        "drone_locations": drone_locs, 
                        "drone_batteries": drone_batts, 
                        "drone_states": drone_states,
                        "t": t
                    },
                    {}
                )
                print(f"    ⏱️ timestep {t} actions: {actions}")
                for i, act in enumerate(actions):
                    if act[0] == "move":
                        dx, dy = act[1]
                        x, y = drone_locs[i]
                        drone_locs[i] = (x + dx, y + dy)
                    elif act[0] == "charge":
                        drone_batts[i] = (100, 100)

                print(f"    📍 Drone positions after timestep {t}: {drone_locs}")
                print(f"    🔋 Drone batteries after timestep {t}: {drone_batts}")
            cluster_data.append({
                "cid": cid,
                "color": color_map[cid % len(color_map)],
                "stations": stations_in_cluster,
                "ground_sensors": ground_sensors,
                "charging_stations": stations_in_cluster,
                "init_drone_positions": drone_init_output,
                "final_drone_positions": drone_locs,
                "polygons": polygons,
            })

            print(f"✅ Finished simulation for cluster {cid}")
            
        self.cluster_data = cluster_data
        return strategy_instances, cluster_data

    def plot_clusters(self, title="BFS Clusters (final drone positions)", figsize=(8,8)):
        """
        Plot all cluster data after run_clusters has been called.
        """
        if not self.cluster_data:
            print("[!] No cluster data to plot. Call run_clusters() first.")
            return

        plt.figure(figsize=figsize)
        plt.title(title)

        for cinfo in self.cluster_data:
            cid = cinfo["cid"]
            color = cinfo["color"]

            plt.scatter(*zip(*cinfo["stations"]), c=color, marker='o', s=80, label=f"Cluster {cid} stations")
            plt.scatter(*zip(*cinfo["ground_sensors"]), c=color, marker='s', s=100, edgecolors='black')
            plt.scatter(*zip(*cinfo["charging_stations"]), c=color, marker='*', s=140, edgecolors='black')
            plt.scatter(*zip(*cinfo["init_drone_positions"]), c=color, marker='D', s=80, edgecolors='black')
            plt.scatter(*zip(*cinfo["final_drone_positions"]), c=color, marker='D', s=80, edgecolors='red')

            for poly in cinfo["polygons"]:
                x, y = poly.exterior.xy
                plt.fill(x, y, alpha=0.2, facecolor=color, edgecolor='black')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', 'box')
        plt.tight_layout()
        plt.show()
##############################################################################
# Example usage
##############################################################################
if __name__ == "__main__":
    # Example stations
    stations = [
        (2,2), (3,2), (4,4),
        (10,10), (11,10),
        (22,22), (23,22)
    ]
    drone_battery = 6.0

    # Suppose we have 3 clusters from BFS, so we specify # drones for each cluster:
    # e.g. cluster 0 => 2 drones, cluster 1 => 1 drone, cluster 2 => 2 drones
    drones_per_cluster = [2, 1, 2]
    
    # Import strategy classes
    from Strategy import RandomSensorPlacementStrategy, LoggedDroneRoutingStrategy, DroneRoutingOptimizationModelReuseIndex
    
    # Create a wrapper around the strategy class
    wrapper = DroneRoutingStrategyClusterWrapper(DroneRoutingOptimizationModelReuseIndex)
    
    # Run the clusters, which returns strategy instances and cluster data
    strategy_instances, cluster_data = wrapper.run_clusters(
        charging_stations=stations, 
        drone_battery=drone_battery, 
        drones_per_cluster=drones_per_cluster,
        timesteps=2,
        half_extent=drone_battery / 2.0,
        total_ground_sensors=3,
        SensorStrategyClass=RandomSensorPlacementStrategy,
        custom_init_params={
            "burnmap_filename": "MinimalDataset/0001/burn_map.npy",
            "call_every_n_steps": 5,
            "optimization_horizon": 20,
            "reevaluation_step": 15
        }
    )

    print(strategy_instances, cluster_data)
    # Finally, plot the clusters (including station locations and final drone positions)
    wrapper.plot_clusters(
        title="Example BFS Clusters and Drone Positions", 
        figsize=(8,8)
    )