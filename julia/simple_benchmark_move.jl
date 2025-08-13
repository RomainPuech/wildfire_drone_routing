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

# Test array of length ~60
test_array = collect(1:60)

println("Simple Move Element Performance Comparison")
println("=" ^ 45)
println("Array length: $(length(test_array))")
println()

# Test a few specific moves
test_cases = [
    (10, 20),  # Move from middle to middle
    (1, 30),   # Move from start to middle  
    (30, 1),   # Move from middle to start
    (60, 1),   # Move from end to start
    (1, 60),   # Move from start to end
]

for (i, j) in test_cases
    println("Move element from position $i to position $j:")
    
    # Time approach 1
    t1 = @elapsed move_element_approach1(test_array, i, j)
    # Time approach 2  
    t2 = @elapsed move_element_approach2(test_array, i, j)
    
    println("  Approach 1 (copy/deleteat!/insert!): $(round(t1*1000, digits=6)) ms")
    println("  Approach 2 (move_element function): $(round(t2*1000, digits=6)) ms")
    println("  Speedup: $(round(t1/t2, digits=2))x")
    println()
end

# Test with random moves
println("Random moves (1000 iterations):")
Random.seed!(42)

# Time 1000 random moves for approach 1
t1_total = @elapsed begin
    local arr1 = copy(test_array)
    for _ in 1:1000
        i = rand(1:60)
        j = rand(1:60)
        if i != j
            arr1 = move_element_approach1(arr1, i, j)
        end
    end
end

# Time 1000 random moves for approach 2
t2_total = @elapsed begin
    local arr2 = copy(test_array)
    for _ in 1:1000
        i = rand(1:60)
        j = rand(1:60)
        if i != j
            arr2 = move_element_approach2(arr2, i, j)
        end
    end
end

println("  Approach 1 total time: $(round(t1_total*1000, digits=3)) ms")
println("  Approach 2 total time: $(round(t2_total*1000, digits=3)) ms")
println("  Average per move - Approach 1: $(round(t1_total*1000/1000, digits=6)) ms")
println("  Average per move - Approach 2: $(round(t2_total*1000/1000, digits=6)) ms")
println("  Overall speedup: $(round(t1_total/t2_total, digits=2))x")
