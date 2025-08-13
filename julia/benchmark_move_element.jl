using BenchmarkTools
using Random

# Approach 1: Using copy, deleteat!, and insert!
function move_element_approach1(vec, i, j)
    new_position = copy(vec)
    customer = new_position[i]
    deleteat!(new_position, i)
    insert!(new_position, j > i ? j-1 : j, customer)
    return new_position
end

# Approach 2: Using the move_element function
function move_element_approach2(vec, i, j)
    n = length(vec)
    new_vec = Vector{eltype(vec)}(undef, n)
    customer = vec[i]
    
    # Adjust j if it's after the removed element
    target_j = j > i ? j - 1 : j
    
    new_idx = 1
    for old_idx in 1:n
        if old_idx == i
            continue  # Skip the element we're moving
        end
        
        if new_idx == target_j
            new_vec[target_j] = customer
            new_idx += 1
        end
        
        new_vec[new_idx] = vec[old_idx]
        new_idx += 1
    end
    
    # If target position is at the end
    if target_j == n
        new_vec[n] = customer
    end
    
    return new_vec
end

# Benchmark functions for random moves
function benchmark_random_moves_approach1(test_array)
    arr = copy(test_array)
    for _ in 1:1000
        i = rand(1:60)
        j = rand(1:60)
        if i != j
            arr = move_element_approach1(arr, i, j)
        end
    end
    return arr
end

function benchmark_random_moves_approach2(test_array)
    arr = copy(test_array)
    for _ in 1:1000
        i = rand(1:60)
        j = rand(1:60)
        if i != j
            arr = move_element_approach2(arr, i, j)
        end
    end
    return arr
end

# Test function to verify both approaches give the same result
function test_correctness()
    println("Testing correctness...")
    
    # Test with array of length ~60
    test_array = collect(1:60)
    
    # Test a few different moves
    test_cases = [
        (10, 20),  # Move from middle to middle
        (1, 30),   # Move from start to middle
        (30, 1),   # Move from middle to start
        (60, 1),   # Move from end to start
        (1, 60),   # Move from start to end
        (15, 45),  # Move within second half
    ]
    
    for (i, j) in test_cases
        result1 = move_element_approach1(test_array, i, j)
        result2 = move_element_approach2(test_array, i, j)
        
        if result1 != result2
            println("ERROR: Results differ for move ($i, $j)")
            println("Approach 1: $result1")
            println("Approach 2: $result2")
            return false
        else
            println("✓ Move ($i, $j): Results match")
        end
    end
    
    println("All correctness tests passed!")
    return true
end

# Benchmark function
function run_benchmarks()
    println("Running performance benchmarks...")
    
    # Create test array of length ~60
    test_array = collect(1:60)
    
    # Define test cases for benchmarking
    test_cases = [
        (10, 20),  # Move from middle to middle
        (1, 30),   # Move from start to middle
        (30, 1),   # Move from middle to start
        (60, 1),   # Move from end to start
        (1, 60),   # Move from start to end
    ]
    
    println("\nBenchmarking individual moves:")
    println("=" ^ 50)
    
    for (i, j) in test_cases
        println("\nMove element from position $i to position $j:")
        
        # Benchmark approach 1
        b1 = @benchmark move_element_approach1($test_array, $i, $j)
        println("  Approach 1 (copy/deleteat!/insert!):")
        println("    Minimum time: $(minimum(b1.times) / 1_000_000) ms")
        println("    Median time:  $(median(b1.times) / 1_000_000) ms")
        println("    Mean time:    $(mean(b1.times) / 1_000_000) ms")
        
        # Benchmark approach 2
        b2 = @benchmark move_element_approach2($test_array, $i, $j)
        println("  Approach 2 (move_element function):")
        println("    Minimum time: $(minimum(b2.times) / 1_000_000) ms")
        println("    Median time:  $(median(b2.times) / 1_000_000) ms")
        println("    Mean time:    $(mean(b2.times) / 1_000_000) ms")
        
        # Calculate speedup
        speedup = mean(b1.times) / mean(b2.times)
        println("  Speedup: $(round(speedup, digits=2))x")
    end
    
    # Benchmark with random moves (more realistic scenario)
    println("\n" * "=" ^ 50)
    println("Benchmarking random moves (1000 iterations):")
    
    Random.seed!(42)  # For reproducible results
    
    # Benchmark random moves
    b1_rand = @benchmark benchmark_random_moves_approach1($test_array)
    b2_rand = @benchmark benchmark_random_moves_approach2($test_array)
    
    println("  Approach 1 (copy/deleteat!/insert!):")
    println("    Total time for 1000 moves: $(mean(b1_rand.times) / 1_000_000) ms")
    println("    Average per move: $(mean(b1_rand.times) / 1_000_000 / 1000) ms")
    
    println("  Approach 2 (move_element function):")
    println("    Total time for 1000 moves: $(mean(b2_rand.times) / 1_000_000) ms")
    println("    Average per move: $(mean(b2_rand.times) / 1_000_000 / 1000) ms")
    
    speedup_rand = mean(b1_rand.times) / mean(b2_rand.times)
    println("  Overall speedup: $(round(speedup_rand, digits=2))x")
end

# Main execution
function main()
    println("Move Element Performance Benchmark")
    println("=" ^ 40)
    
    # First test correctness
    if !test_correctness()
        println("Correctness test failed! Aborting benchmarks.")
        return
    end
    
    # Then run benchmarks
    run_benchmarks()
    
    println("\n" * "=" ^ 40)
    println("Benchmark completed!")
end

# Run the benchmark
main()
