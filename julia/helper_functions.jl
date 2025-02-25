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