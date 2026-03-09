# Test to see kernel coverage values near the center
# Run this from the julia directory: julia test_kernel_values.jl

using Statistics

println("="^70)
println("KERNEL COVERAGE VALUES ANALYSIS")
println("="^70)
println()

function create_simple_kernel(kernel_size::Int)
    kernel = Dict{Tuple{Int,Int}, Float64}()
    for dx in -kernel_size:kernel_size
        for dy in -kernel_size:kernel_size
            dist = max(abs(dx), abs(dy))
            if dist <= kernel_size
                weight = max(0.0, 1.0 - dist / (kernel_size + 1))
                if weight > 0.01
                    kernel[(dx, dy)] = weight
                end
            end
        end
    end
    return kernel
end

# Test with different kernel sizes
kernel_sizes = [5, 6, 8, 10, 12]

for kernel_size in kernel_sizes
    println("="^70)
    println("Kernel Size: $kernel_size")
    println("="^70)
    
    kernel = create_simple_kernel(kernel_size)
    
    # Show values for cells near the center (dx, dy from charging station)
    println("Coverage values for cells near charging station (dx, dy from station):")
    println()
    println("Distance from station | Coverage weight")
    println("-"^70)
    
    # Group by L-infinity distance
    for dist in 0:min(5, kernel_size)
        cells_at_dist = []
        for dx in -kernel_size:kernel_size
            for dy in -kernel_size:kernel_size
                if max(abs(dx), abs(dy)) == dist
                    weight = get(kernel, (dx, dy), 0.0)
                    if weight > 0.01
                        push!(cells_at_dist, ((dx, dy), weight))
                    end
                end
            end
        end
        
        if length(cells_at_dist) > 0
            avg_weight = mean([w for (_, w) in cells_at_dist])
            min_weight = minimum([w for (_, w) in cells_at_dist])
            max_weight = maximum([w for (_, w) in cells_at_dist])
            
            println("Distance $dist (L∞):")
            println("  Average coverage: $(round(avg_weight, digits=4))")
            println("  Min coverage: $(round(min_weight, digits=4))")
            println("  Max coverage: $(round(max_weight, digits=4))")
            println("  Number of cells: $(length(cells_at_dist))")
            
            # Show a few examples
            if dist <= 3
                println("  Examples:")
                for ((dx, dy), w) in cells_at_dist[1:min(5, length(cells_at_dist))]
                    println("    Cell at ($dx, $dy): $(round(w, digits=4))")
                end
            end
            println()
        end
    end
    
    # Show what happens with multiple drones
    println("Coverage with multiple drones (capped at 1.0):")
    center_weight = get(kernel, (0, 0), 0.0)
    println("  Center cell (0, 0) coverage: $(round(center_weight, digits=4))")
    for n_drones in 1:5
        coverage = min(1.0, center_weight * n_drones)
        println("  With $n_drones drone(s): $(round(coverage, digits=4)) (capped at 1.0)")
    end
    
    # Check nearby cells
    println()
    println("Nearby cells (distance 1-2):")
    for dist in 1:2
        for dx in -dist:dist
            for dy in -dist:dist
                if max(abs(dx), abs(dy)) == dist
                    weight = get(kernel, (dx, dy), 0.0)
                    if weight > 0.01
                        println("  Cell ($dx, $dy) at distance $dist: $(round(weight, digits=4))")
                        for n_drones in 1:5
                            coverage = min(1.0, weight * n_drones)
                            if coverage < 1.0
                                println("    With $n_drones drone(s): $(round(coverage, digits=4))")
                            else
                                println("    With $n_drones drone(s): 1.0 (capped)")
                                break
                            end
                        end
                    end
                end
            end
        end
    end
    
    println()
    println()
end

println("="^70)
println("ANALYSIS")
println("="^70)
println()
println("Key observations:")
println("1. The kernel provides coverage that decreases with distance")
println("2. Cells very close to the station (distance 0-1) may already have high coverage")
println("3. With multiple drones, coverage = min(1.0, kernel_weight * n_drones)")
println("4. If kernel_weight is already high (e.g., 0.8+), adding drones quickly hits the cap")
println("5. This explains why concentration may not help if nearby cells are already well-covered")
println()
