using NPZ


function load_burn_map(filename, static_map=false)
    
    try
        # Read the file
        #println("Loading burn map from $filename")
        burn_map = npzread(filename)
        # if static map, duplicate the data to go from shape (1,N,M) to shape (100,N,M)
        if static_map
            burn_map = repeat(burn_map, outer=(100, 1, 1))
        end
        return burn_map
    catch e
        error("Error loading burn map: $e")
    end
end

function L_inf_distance(a,b)
    """
    Returns the L-infinity distance between a and b in R^n
    """
    return maximum(abs.(a .- b))
end


function L_1_distance(a,b)
    """
    Returns the L-1 distance between a and b in R^n
    """
    return sum(abs.(a .- b))
end

function L_2_distance(a,b)
    """
    Returns the L-2 distance between a and b in R^n
    """
    return sqrt(sum((a .- b).^2))
end

function compute_distance(point1, point2, metric="linf")
    """
    Computes the distance between two points using the specified metric.
    
    Arguments:
    - point1, point2: Points to compute distance between
    - metric: String indicating which distance metric to use ("l1", "l2", or "linf")
    
    Returns:
    - The distance between the points according to the specified metric
    """
    if metric == "linf"
        return L_inf_distance(point1, point2)
    elseif metric == "l1"
        # Euclidean distance
        return L_1_distance(point1, point2)
    elseif metric == "l2"
        # Manhattan distance
        return L_2_distance(point1, point2)
    else
        error("Invalid metric: $metric")
    end
end

function L_inf_neighbors(i, n=1, I=nothing)
    """
    Returns all points with L-infinity distance less than or equal to n from point i,
    EXCLUDING the point i itself, intersected with feasible set I if provided.
    
    Arguments:
    - i: The center point (tuple or array)
    - n: Maximum L-infinity distance for neighbors
    - I: Optional feasible set to intersect with
    
    Returns:
    - List of points within L-infinity distance n of point i, excluding i itself
    """
    dim = length(i)
    neighbors_list = []
    
    # Generate all possible combinations of moves in n dimensions
    ranges = [(-n):n for _ in 1:dim]
    for moves in Iterators.product(ranges...)
        # Skip the point itself (when all moves are 0)
        if any(m != 0 for m in moves)
            point = [i[j] + moves[j] for j in 1:dim]
            if I === nothing || Tuple(point) in I
                push!(neighbors_list, Tuple(point))
            end
        end
    end
    
    return neighbors_list
end

function L_inf_neighbors_and_point(i, I=nothing)
    """
    Returns the L-infinity norm-neighbors of i in Z^n, intersected with feasible set I if provided
    (returns the feasible cells directly around i)
    """
    n = length(i)
    neighbors_list = []
    
    # Generate all possible combinations of -1, 0, 1 in n dimensions
    for moves in Iterators.product(fill((-1,0,1), n)...)
        point = [i[j] + moves[j] for j in 1:n]
        if I === nothing || point in I # if the point belongs to the original set I
            push!(neighbors_list, Tuple(i[j] + moves[j] for j in 1:n))
        end
    end
    
    return neighbors_list
end

function L_1_neighbors_and_point(i, I=nothing)
    """
    Returns the L1 norm-neighbors (Manhattan distance) of i in Z^n, intersected with feasible set I if provided.
    Only includes points with Manhattan distance <= 1 from i.
    """
    n = length(i)
    neighbors_list = []
    
    # Add the center point itself
    center_point = Tuple(i)
    if I === nothing || center_point in I
        push!(neighbors_list, center_point)
    end
    
    # For each dimension, add the two points where we move +1 or -1 in that dimension only
    for dim in 1:n
        for move in [-1, 1]
            point = collect(i)  # Convert to array for mutation
            point[dim] += move  # Move in only this dimension
            point_tuple = Tuple(point)
            
            if I === nothing || point_tuple in I
                push!(neighbors_list, point_tuple)
            end
        end
    end
    
    return neighbors_list
end

function L_2_neighbors_and_point(i, I=nothing)
    """
    Returns the L2 norm-neighbors (Euclidean distance) of i in Z^n, intersected with feasible set I if provided.
    Only includes points with Euclidean distance <= 1 from i.
    """
    n = length(i)
    neighbors_list = []
    
    # Generate all possible combinations of -1, 0, 1 in n dimensions
    for moves in Iterators.product(fill((-1,0,1), n)...)
        # Calculate Euclidean distance as sqrt(sum of squares)
        euclidean_dist = sqrt(sum(moves.^2))
        
        # Only include points with Euclidean distance <= 1
        if euclidean_dist <= 1
            point = [i[j] + moves[j] for j in 1:n]
            if I === nothing || point in I
                push!(neighbors_list, Tuple(point))
            end
        end
    end
    
    return neighbors_list
end

function neighbors_and_point(i, I=nothing, metric="linf", radius=1)
    if metric == "linf"
        return L_inf_neighbors_and_point(i, I)
    elseif metric == "l1"
        return L_1_neighbors_and_point(i, I)
    elseif metric == "l2"
        return L_2_neighbors_and_point(i, I)
    end
end

function get_drone_gridpoints(charging_stations, n, I)
    """
    Returns the set of points covered by charging stations within L-infinity distance n.

    Arguments:
    - charging_stations: List of tuples representing the (x, y) locations of charging stations.
    - n: Maximum L-infinity distance for coverage.
    - grid_points: Set of all possible points in the region (e.g., a list of (x, y) tuples).

    Returns:
    - Set of (x, y) points that are within L-infinity distance n from any charging station.
    """
    covered_points = Set()
    for i in I
        for c in charging_stations
            if L_inf_distance(i, c) <= n
                push!(covered_points, i)
                break  # No need to check other stations once it's covered
            end
        end
    end
    return covered_points
end

function get_drone_gridpoints_using_neighbors(charging_stations, n, I)
    """
    Returns the set of points covered by charging stations within L-infinity distance n.

    Arguments:
    - charging_stations: List of tuples representing the locations of charging stations.
    - n: Maximum L-infinity distance for coverage.
    - I: Set of all possible points in the region.

    Returns:
    - Set of points that are within L-infinity distance n from any charging station.
    """
    covered_points = Set()
    
    # Use the neighbors function to find all points within distance n of each station
    for station in charging_stations
        # Get neighbors (excluding the station itself)
        station_coverage = L_inf_neighbors(station, n, I)
        
        # Add the station itself if it's in the feasible set
        if station in I
            push!(covered_points, station)
        end
        
        # Add all neighbors to covered points
        union!(covered_points, station_coverage)
    end
    
    return covered_points
end


function phi(x,y)
    return L_inf_distance(x, y) <= 4 ? 1 : 0
end

function test()
    println("test")
end

# We can implement this using a KD-tree for efficiency but the overhead is too high for small inputs
function closest_distances(neighbors, points, metric="linf")
    """
    Returns a list of the minimum distances from each point in 'points' to any point in 'neighbors'.
    
    Arguments:
    - neighbors: List of tuples or arrays representing points in a grid (e.g., [(x1,y1), (x2,y2), ...])
    - points: List of tuples or arrays representing points in the same grid
    - metric: Distance metric to use ("l1", "l2", or "linf")
    
    Returns:
    - A list of distances, where each element is the minimum distance from the corresponding point in 'points'
      to any point in 'neighbors' using the specified metric
    """
    distances = []
    
    for point in points
        min_dist = Inf
        for neighbor in neighbors
            dist = compute_distance(point, neighbor, metric)
            if dist < min_dist
                min_dist = dist
            end
        end
        push!(distances, min_dist)
    end
    
    return distances
end

function closest_distances_tuple_index(neighbors, points, metric="linf")
    """
    Returns a list of the minimum distances from each point in 'points' to any point in 'neighbors'.
    
    Arguments:
    - neighbors: List of tuples or arrays representing points in a grid (e.g., [(x1,y1), (x2,y2), ...])
    - points: List of tuples or arrays representing points in the same grid
    - metric: Distance metric to use ("l1", "l2", or "linf")
    
    Returns:
    - A list of distances, where each element is the minimum distance from the corresponding point in 'points'
      to any point in 'neighbors' using the specified metric
    """
    distances = Dict()
    
    for point in points
        min_dist = Inf
        for neighbor in neighbors
            dist = compute_distance(point, neighbor, metric)
            if dist < min_dist
                min_dist = dist
            end
        end
        distances[point] = min_dist
    end
    
    return distances
end

function closest_distance(neighbors, point, metric="linf")
    """
    Returns the minimum distance from 'point' to any point in 'neighbors'.
    
    Arguments:
    - neighbors: List of tuples or arrays representing points in a grid (e.g., [(x1,y1), (x2,y2), ...])
    - point: Tuple or array representing a point in the same grid
    - metric: Distance metric to use ("l1", "l2", or "linf")
    
    Returns:
    - The minimum distance from 'point' to any point in 'neighbors' using the specified metric
    """
    min_dist = Inf
    for neighbor in neighbors
        dist = compute_distance(point, neighbor, metric)
        if dist < min_dist
            min_dist = dist
        end
    end
    return min_dist
end