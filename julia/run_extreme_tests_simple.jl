# Simple test runner that runs tests one at a time with visible progress
# Run with: julia run_extreme_tests_simple.jl

include("test_extreme_scenarios.jl")

# Just run the original 3 scenarios for quick testing
println("Running quick test with original 3 scenarios only...")
println()

for (test_idx, (scenario_name, create_fn, N, M, N_grounds, N_charging, n_drones, max_battery)) in enumerate(test_configs_original)
    println("="^70)
    println("TEST $test_idx/3: $scenario_name")
    println("="^70)
    flush(stdout)
    
    result = run_extreme_test(scenario_name, create_fn, N, M, N_grounds, N_charging, n_drones, max_battery)
    
    println("✓ Test $test_idx completed")
    println()
    flush(stdout)
end

println("="^70)
println("All original tests completed!")
println("="^70)
