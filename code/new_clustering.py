import math
from collections import deque
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from Strategy import DroneRoutingStrategy

def get_wrapped_strategy(BaseStrategy, charging_stations, drone_battery, drones_per_cluster):
    class ClusteredDroneStrategyWrapped(BaseStrategy):
        def __init__(self,automatic_initialization_parameters:dict, custom_initialization_parameters:dict):
            self.strategy_instances = []
            self.drones_per_cluster = drones_per_cluster
            self.total_drones = sum(drones_per_cluster)

            self.clusters = self.find_clusters(charging_stations, drone_battery)
            print(f"[init] Number of clusters: {len(self.clusters)}")
            for i, cluster in enumerate(self.clusters):
                print(f"  Cluster {i}: {cluster}")
            self.cluster_data = []

            # global bounding box to compute ground sensors
            all_polygons = []
            for stations in self.clusters:
                all_polygons += self.get_cluster_boundary_boxes(stations, half_extent)
            N, M, min_x, min_y = self.get_bounding_grid_size(all_polygons)


            for cid, stations in enumerate(self.clusters):
                print(f"\n🚀 Running cluster {cid} with {len(stations)} charging stations and {drones_per_cluster[cid]} drones")
                polygons = self.get_cluster_boundary_boxes(stations, half_extent)
                cluster_poly = unary_union(polygons)
                N, M, min_x, min_y = self.get_bounding_grid_size(polygons)
                print(f"  🧱 Bounding grid: {N} x {M}, origin: ({min_x}, {min_y})")
                strat = StrategyClass(auto_params, custom_init_params)
                self.strategy_instances.append(strat)
                print(f"  ✅ Strategy initialized for cluster {cid}")

                self.cluster_data.append({
                    "cid": cid,
                    "stations": stations,
                    "ground_sensors": ground_sensors,
                    "charging_stations": stations,
                    "polygons": polygons,
                    "color": ["blue", "orange", "green", "red", "purple", "cyan", "magenta", "gray"][cid % 8]
                })

            self.initialized = False
            self.initial_positions = []
            self.initial_states = []

        def find_clusters(self, charging_stations, drone_battery):
            radius = drone_battery / 2.0
            n = len(charging_stations)
            adj = [[] for _ in range(n)]

            for i in range(n):
                for j in range(i+1, n):
                    if math.hypot(charging_stations[i][0] - charging_stations[j][0], charging_stations[i][1] - charging_stations[j][1]) <= radius:
                        adj[i].append(j)
                        adj[j].append(i)

            cluster_labels = [-1] * n
            cluster_id = 0
            for start in range(n):
                if cluster_labels[start] == -1:
                    queue = deque([start])
                    cluster_labels[start] = cluster_id
                    while queue:
                        node = queue.popleft()
                        for neighbor in adj[node]:
                            if cluster_labels[neighbor] == -1:
                                cluster_labels[neighbor] = cluster_id
                                queue.append(neighbor)
                    cluster_id += 1

            clusters = [[] for _ in range(cluster_id)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(charging_stations[i])
            return clusters

        def get_cluster_boundary_boxes(self, stations, half_extent):
            boxes = []
            for x, y in stations:
                box = Polygon([
                    (x - half_extent, y - half_extent),
                    (x + half_extent, y - half_extent),
                    (x + half_extent, y + half_extent),
                    (x - half_extent, y + half_extent),
                ])
                boxes.append(box)
            unioned = unary_union(boxes)
            if unioned.is_empty:
                return []
            if unioned.geom_type == "Polygon":
                return [unioned]
            return list(unioned)

        def get_bounding_grid_size(self, polygons):
            if not polygons:
                return (1, 1, 0, 0)
            bounds = unary_union(polygons).bounds
            minx, miny, maxx, maxy = map(float, bounds)
            return (
                math.ceil(maxx) - math.floor(minx) + 1,
                math.ceil(maxy) - math.floor(miny) + 1,
                math.floor(minx),
                math.floor(miny)
            )

        def get_initial_drone_locations(self):
            positions = []
            states = []
            
            print(f"\n📍 [ClusteredDroneStrategyWrapper] Fetching initial drone locations for {len(self.strategy_instances)} clusters...")

            for i, strat in enumerate(self.strategy_instances):
                print(f"\n📦 Cluster {i}: Calling strategy to get initial positions and states...")
                pos, state = strat.get_initial_drone_locations()

                for d, (p, s) in enumerate(zip(pos, state)):
                    print(f"   🛰️ Drone {d}: {s} at {p}")

                positions.extend(pos)
                states.extend(state)

            self.initialized = True
            self.initial_positions = positions
            self.initial_states = states

            print(f"\n✅ [ClusteredDroneStrategyWrapper] Combined total drones: {len(positions)}")
            return positions, states

        def next_actions(self, automatic_step_parameters, custom_step_parameters):
            t = automatic_step_parameters['t']
            print(f"\n⏱️ [ClusteredDroneStrategyWrapper] Timestep {t} - computing actions...")

            if not self.initialized:
                raise RuntimeError("Must call get_initial_drone_locations() first.")

            actions = []
            idx = 0

            for i, (count, strat) in enumerate(zip(self.drones_per_cluster, self.strategy_instances)):
                sliced_params = {
                    "drone_locations": automatic_step_parameters["drone_locations"][idx:idx+count],
                    "drone_batteries": automatic_step_parameters["drone_batteries"][idx:idx+count],
                    "drone_states": automatic_step_parameters["drone_states"][idx:idx+count],
                    "t": t
                }

                print(f"\n📡 Cluster {i} handling drones {idx} to {idx+count-1}")
                for d, (loc, st) in enumerate(zip(sliced_params['drone_locations'], sliced_params['drone_states'])):
                    print(f"   🛰️ Drone {idx + d}: {st} at {loc}")

                cluster_actions = strat.next_actions(sliced_params, custom_step_parameters)
                actions.extend(cluster_actions)

                print(f"   🧠 Actions from cluster {i}:")
                for d, act in enumerate(cluster_actions):
                    print(f"     ↪️ Drone {idx + d}: {act}")

                idx += count

            print(f"\n✅ [ClusteredDroneStrategyWrapper] Combined actions for timestep {t}: {actions}")
            return actions

        def plot_clusters(self, title="Clustered Drone Layout", figsize=(8,8)):
            if not self.cluster_data:
                print("No cluster data available.")
                return
            plt.figure(figsize=figsize)
            plt.title(title)
            for cinfo in self.cluster_data:
                cid = cinfo["cid"]
                color = cinfo["color"]
                if cinfo["stations"]:
                    plt.scatter(*zip(*cinfo["stations"]), c=color, marker='o', s=80, label=f"Cluster {cid} stations")
                if cinfo["ground_sensors"]:
                    plt.scatter(*zip(*cinfo["ground_sensors"]), c=color, marker='s', s=100, edgecolors='black')
                if cinfo["charging_stations"]:
                    plt.scatter(*zip(*cinfo["charging_stations"]), c=color, marker='*', s=140, edgecolors='black')
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

    # def run(self, timesteps=5, verbose=True):

    #     print(f"\n▶️ Running clustered strategy for {timesteps} timesteps")

    #     locs, states = self.get_initial_drone_locations()
    #     batts = [(100, 100)] * len(locs)

    #     for t in range(timesteps):
    #         if verbose:
    #             print(f"\n⏱️ timestep {t}")
    #         actions = self.next_actions({
    #             "drone_locations": locs,
    #             "drone_batteries": batts,
    #             "drone_states": states,
    #             "t": t
    #         }, {})

    #         new_locs = []
    #         new_states = []
    #         for i, action in enumerate(actions):
    #             x, y = locs[i]
    #             if action[0] == "move":
    #                 dx, dy = action[1]
    #                 new_locs.append((x + dx, y + dy))
    #                 new_states.append("fly")
    #             elif action[0] == "fly":
    #                 new_locs.append(action[1])
    #                 new_states.append("fly")
    #             elif action[0] == "charge":
    #                 new_locs.append((x, y))
    #                 new_states.append("charge")
    #             else:
    #                 new_locs.append((x, y))
    #                 new_states.append("fly")

    #             if verbose:
    #                 moved = new_locs[-1] != (x, y)
    #                 print(f"  🚁 Drone {i} {'moved to' if moved else 'stayed at'} {new_locs[-1]} [{action[0]}]")

    #         locs = new_locs
    #         states = new_states
    #         batts = [(100, 100)] * len(locs)  # reset battery for simplicity

    #     print("\n✅ Simulation finished.")


    return ClusteredDroneStrategyWrapper


if __name__ == "__main__":
    from wrappers import wrap_log_drone_strategy
    from Strategy import RandomSensorPlacementStrategy, DroneRoutingOptimizationModelReuseIndex

    LoggedDroneRoutingOptimizationModelReuseIndex = wrap_log_drone_strategy(DroneRoutingOptimizationModelReuseIndex)

    # example charging station layout
    stations = [
        (2, 2), (3, 2), (4, 4),       # cluster 0
        (10, 10), (11, 10),           # cluster 1
        (22, 22), (23, 22), (23, 23)  # cluster 2
    ]
    drone_battery = 6.0

    # one entry per cluster (determined automatically by BFS)
    drones_per_cluster = [2, 1, 2]  # must match # of clusters expected

    wrapped_strategy = ClusteredDroneStrategyWrapper(
        StrategyClass=LoggedDroneRoutingOptimizationModelReuseIndex,
        charging_stations=stations,
        drone_battery=drone_battery,
        drones_per_cluster=drones_per_cluster,
        timesteps=10,
        half_extent=drone_battery / 2.0,
        total_ground_sensors=3,
        SensorStrategyClass=RandomSensorPlacementStrategy,
        custom_init_params={
            "burnmap_filename": "MinimalDataset/0001/burn_map.npy",
            "call_every_n_steps": 5,
            "optimization_horizon": 20,
            "reevaluation_step": 10
        }
    )
    # do we need .run???
    wrapped_strategy.run(timesteps=5)

    # optional debug plot
    wrapped_strategy.plot_clusters()
