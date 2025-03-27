import math
import matplotlib.pyplot as plt
from collections import deque

def find_clusters(charging_stations, drone_battery):
    """
    charging_stations : list of (x, y) positions
    drone_battery     : float, total battery range
                       We define the 'connection radius' = drone_battery / 2
    Returns:
      cluster_labels: list of cluster indices (same length as charging_stations).
                     cluster_labels[i] is the cluster ID of station i.
      bounding_boxes: list of (min_x, max_x, min_y, max_y) for each cluster.
    """

    n = len(charging_stations)
    radius = drone_battery / 2.0

    # --- 1) Build adjacency (who is within 'radius' of whom) ---
    adjacency_list = [[] for _ in range(n)]

    for i in range(n):
        x1, y1 = charging_stations[i]
        for j in range(i + 1, n):
            x2, y2 = charging_stations[j]
            dist = math.hypot(x2 - x1, y2 - y1)
            # If these two stations lie within radius, connect them in the graph
            if dist <= radius:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    # --- 2) Find connected components (clusters) via BFS or DFS ---
    cluster_labels = [-1] * n  # -1 means unvisited
    current_cluster = 0

    for start_node in range(n):
        if cluster_labels[start_node] == -1:
            # BFS from this node
            queue = deque([start_node])
            cluster_labels[start_node] = current_cluster

            while queue:
                node = queue.popleft()
                for neighbor in adjacency_list[node]:
                    if cluster_labels[neighbor] == -1:
                        cluster_labels[neighbor] = current_cluster
                        queue.append(neighbor)

            current_cluster += 1

    # --- 3) For each cluster, compute bounding box among its stations ---
    bounding_boxes = []
    num_clusters = current_cluster  # BFS increments each time we find a new cluster

    for c in range(num_clusters):
        # All stations in cluster c
        indices_in_cluster = [i for i, lbl in enumerate(cluster_labels) if lbl == c]
        # Extract x and y coordinates for just this cluster
        xs = [charging_stations[i][0] for i in indices_in_cluster]
        ys = [charging_stations[i][1] for i in indices_in_cluster]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bounding_boxes.append((min_x, max_x, min_y, max_y))

    return cluster_labels, bounding_boxes

def plot_clusters(charging_stations, cluster_labels, bounding_boxes):
    """
    charging_stations : list of (x, y) positions
    cluster_labels    : list of cluster IDs for each station
    bounding_boxes    : list of (min_x, max_x, min_y, max_y) for each cluster
    """
    plt.figure(figsize=(8, 6))
    plt.title("Charging Station Clusters")

    num_clusters = max(cluster_labels) + 1
    colors = plt.cm.get_cmap("tab20", num_clusters)

    for i, (x, y) in enumerate(charging_stations):
        cluster_id = cluster_labels[i]
        plt.scatter(x, y, c=[colors(cluster_id)], s=60, label=f"Cluster {cluster_id}" if i == cluster_id else "")

    # Plot bounding boxes in different colors
    for cid, box in enumerate(bounding_boxes):
        min_x, max_x, min_y, max_y = box
        # Draw rectangle
        plt.plot([min_x, min_x, max_x, max_x, min_x],
                 [min_y, max_y, max_y, min_y, min_y],
                 color=colors(cid), linewidth=2, label=f"Box of Cluster {cid}")

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(loc="best", ncol=2)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # EXAMPLE USAGE:
    stations = [(2,1), (2,2), (4,4), (10,9), (11,9), (10,11)]
    drone_battery = 6.0  # e.g., if the drone can travel distance=6 total,
                         # radius = 3 => can do a station→station→station round-trip if dist ≤ 3

    labels, boxes = find_clusters(stations, drone_battery)
    
    # Print results
    for i, (sx, sy) in enumerate(stations):
        print(f"Station {i} at ({sx},{sy}) => Cluster {labels[i]}")
    for c, (xmin, xmax, ymin, ymax) in enumerate(boxes):
        print(f"Cluster {c} bounding box: x in [{xmin}, {xmax}], y in [{ymin}, {ymax}]")

    # Display everything
    plot_clusters(stations, labels, boxes)
