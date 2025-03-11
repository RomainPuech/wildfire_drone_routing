function load_burn_map(filename)
    
    try
        # Read the file
        #println("Loading burn map from $filename")
        burn_map = npzread(filename)
        #println("Burn map loaded")
        return burn_map
    catch e
        error("Error loading burn map: $e")
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
