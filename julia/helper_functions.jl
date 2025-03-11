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

function L_inf_distance(a,b)
    """
    Returns the L-infinity distance between a and b in R^n
    """
    return maximum(abs.(a .- b))
end

function neighbors(i, I=nothing)
    """
    Returns the L-infinity norm-neighbors of i in Z^n, intersected with feasible set I if provided
    (returns the feasible cells directly around i)
    """
    n = length(i)
    neighbors_list = []
    
    # Generate all possible combinations of -1, 0, 1 in n dimensions
    for moves in Iterators.product(fill((-1,0,1), n)...)
        if any(m != 0 for m in moves)  # Skip the point itself
            point = [i[j] + moves[j] for j in 1:n]
            if I === nothing || point in I # if the point belongs to the original set I
                push!(neighbors_list, Tuple(i[j] + moves[j] for j in 1:n))
            end
        end
    end
    
    return neighbors_list
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

function phi(x,y)
    return L_inf_distance(x, y) <= 4 ? 1 : 0
end

function test()
    println("test")
end