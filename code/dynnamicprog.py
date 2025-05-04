import heapq
import numpy as np

class ProbaCache:
    """
    Cache for probabilities P(B_{i,t} = 0 | B_{j1,t-1}=0, ..., B_{jk,t-1}=0).
    Keys are tuples (i, t, j1, j2, ..., jk) with js sorted ascending.
    """
    def __init__(self, p, q, neighbors, H):
        self.proba = {(i,0):1.0 for i in p} # we start with no fire
        self.p = p
        self.q = q
        self.neighbors = neighbors
        self.H = H

    def compute_proba(self, H, *args):
        """
        Computes for all i,t: P(B_{i,t} = 0 | B_{j,t-1}=0 for all j in args).
        Uses DP recurrence.
        """
        sorted_args = tuple(sorted(args))

        # check if cache is already computed
        if (0,1) + sorted_args in self.proba:
            print("Cache hit")
            return
        print(f"Computing P(B_{{i,t}} = 0 | B_{{j,t-1}} = 0 for all j in {sorted_args}) for all t = 1,...,{H}")
        # initialize cache
        for i in self.p:
            self.proba[(i,0, *sorted_args)] = 1.0

        # compute all grid points
        C = set(args)
        for t in range(1, H + 1):
            for i in self.p:
                if t == H + 1:
                    self.proba[i,t, *sorted_args] = 0
                else:
                    prod = self.proba[i,t-1, *sorted_args] * (1 - self.p[i])
                    for j in self.neighbors.get(i, []):
                        # override: if j conditioned, alpha_j=1 else use precomputed value
                        if j in C:
                            alpha_j = 1.0
                        else:
                            alpha_j = self.proba[j,t - 1, *sorted_args]
                        prod *= (1 - self.q.get((j, i), 0) * (1 - alpha_j))
                    self.proba[i,t, *sorted_args] = prod
        # print("\n")
        # print(self.proba)
        # print("\n")

    def get_proba(self, i, t, *args):
        sorted_args = tuple(sorted(args))
        key = (i, t) + sorted_args
        print("get_proba", key)
        if key in self.proba:
            return self.proba[key]
        else:
            self.compute_proba(self.H, *args)
            return self.proba[key]

    def __getitem__(self, key):
        # key: (i, t, *args)
        #print("\n")
        #print(self.proba)
        #print("\n")
        return self.get_proba(*key)

    def pretty_print(self, S = []):
        """
        Pretty print the cache under the form of a grid.
        """
        for t in range(self.H + 2):
            print(f"t = {t}")
            for i in self.p:
                print(f"i = {i}: {self.proba[i,t]}")
            print("\n")


def prob_T(i, t, S, cache):
    """
    Compute P(T(S) = T(S,i) = t) for selected set S.
    P(T(S) = t) = P(B_{i,t}=1 | B_{j,t-1} = 0 for all j in S) * \Pi_{k in 1,...,|S|} P(B_{j_k,t-1}=0 | B_{j_l,t-1}=0 for all l > k)
    """
    proba = (1-cache[i, t, *S]) # P(B_{i,t}=1 | B_{j,t-1} = 0 for all j in S)
    for k in range(len(S)):
        proba *= (cache[S[k], t-1, *S[k+1:]]) # BE CAREFUL, HERE IT IS TECHNICALLY | B_k,t-2 = 0, not t-1. However it is equivalent.
    print(f"\033[91mP(T(x,{i}) = T(x) = {t} for S = {S}) = {proba}\033[0m")  # 91m for bright red, 0m to reset color
    return proba



def expected_T(S, p, q, neighbors, H, cache):
    """
    Compute E[T(x)] = sum_{i in S} sum_{t=1}^H t * P(T = T(i)=t).
    """
    total = 0.0
    for i in S:
        for t in range(1, H + 2):
            total += t * prob_T(i, t, S, cache)
    print("E[T(x)] for S =", S, "is", total)
    return total


# SUBMODULAR SO YOU CAN USE VAL OF THE PREVIOUS SELECTION AS A BOUND 
# AND
# instead of computing the marginal gain you can use the marginal gain if you didn't add the previous node: this way you can reuse P(B_{i,t}=0 | B_{j,t-1}=0 for all j in S)
# but first make the algo correct and then try to optimize
# think abt all shortcuts you can take using the submodularity

def lazy_greedy(p, q, neighbors, H, m):
    """
    Lazy greedy algorithm to select up to m nodes maximizing E[T(x)].
    """
    # 1) Precompute burn probabilities and init cache
    cache = ProbaCache(p, q, neighbors, H)
    cache.compute_proba(H)

    # 2) Initialize selection S and current value
    S = []
    current_val = 0.0

    # 3) Build initial heap of marginal gains Δ_j = E[T({j})]
    heap = []
    cache_gain = {}
    for j in p:
        gain = H+1 - expected_T([j], p, q, neighbors, H, cache)
        cache_gain[j] = gain
        heapq.heappush(heap, (gain, j, 0))

    # 4) Greedy selection with lazy updates
    while len(S) < m and heap:
        neg_gain, j, last_size = heapq.heappop(heap)
        if j in S:
            continue
        # If cached gain outdated, recompute true gain
        if last_size != len(S):
            true_gain = expected_T(S + [j], p, q, neighbors, H, cache) - current_val
            cache_gain[j] = true_gain
            heapq.heappush(heap, (-true_gain, j, len(S)))
        else:
            # Accept j
            S.append(j)
            current_val += cache_gain[j]

    return S, current_val, cache


# Example usage

def simulate_fire(p, q, neighbors, H, m, size):
    """
    Simulate the fire and compute the probability of fire spreading to each node.
    """
    grid = np.zeros((H+1, size[0], size[1]))
    for t in range(H):  # Changed from H+1 to H to prevent out of bounds access at t+1
        for i in p:
            if grid[t,i[0],i[1]] == 1:
                grid[t+1,i[0],i[1]] = 1
                for j in neighbors[i]:
                    if grid[t,j[0],j[1]] == 0:
                        die = np.random.rand()
                        if die < q[(i,j)]:
                            grid[t+1,j[0],j[1]] = 1
            else:
                die = np.random.rand()
                if die < p[i]:
                    grid[t+1,i[0],i[1]] = 1
            
    return grid

def simulate_fires_and_compute_probability_grid(p, q, neighbors, H, m, size):
    """
    Simulate the fire and compute the probability of fire at each cell at each time step across scenarios
    """
    N = 100000
    total_grid = np.zeros((H+1, size[0], size[1]))
    for sample in range(N):
        grid = simulate_fire(p, q, neighbors, H, m, size)
        total_grid += grid
    return total_grid / N

def print_grid(grid, title="Grid", color_code="\033[0m"):
    """
    Print a 2D grid in ASCII format with borders and cell values, using specified color.
    """
    print(f"\n{color_code}{title}:\033[0m")
    rows, cols = grid.shape
    # Print top border
    print(color_code + "+" + "-" * (cols * 8 + 1) + "+" + "\033[0m")
    
    for i in range(rows):
        print(color_code + "|" + "\033[0m", end="")
        for j in range(cols):
            print(f" {grid[i,j]:6.3f} ", end="")
        print(color_code + "|" + "\033[0m")
    
    # Print bottom border
    print(color_code + "+" + "-" * (cols * 8 + 1) + "+" + "\033[0m")

if __name__ == "__main__":
    # Create a 5x5 grid with varying probabilities
    p = {(0,0): 0.001, (0,1): 0.002, (0,2): 0.002, (0,3): 0.002, (0,4): 0.002,
         (1,0): 0.005, (1,1): 0.002, (1,2): 0.002, (1,3): 0.004, (1,4): 0.004,
         (2,0): 0.002, (2,1): 0.002, (2,2): 0.002, (2,3): 0.002, (2,4): 0.002,
         (3,0): 0.002, (3,1): 0.002, (3,2): 0.002, (3,3): 0.001, (3,4): 0.002,
         (4,0): 0.002, (4,1): 0.002, (4,2): 0.005, (4,3): 0.002, (4,4): 0.004}
    
    # Create neighbor connections with varying spread probabilities
    q = {}
    for i in p:
        for j in p:
            if i != j:
                # Calculate Manhattan distance
                dist = abs(i[0]-j[0]) + abs(i[1]-j[1])
                if dist <= 2:  # Only allow spread to cells within distance 2
                    # Decrease probability with distance
                    if dist == 1:  # Direct neighbors
                        q[(i,j)] = 0.5
                    else:  # Distance 2 neighbors
                        q[(i,j)] = 0.2
    
    neighbors = {}
    for i in p:
        for j in p:
            if i != j:
                dist = abs(i[0]-j[0]) + abs(i[1]-j[1])
                if dist <= 2:  # Only connect cells within distance 2
                    neighbors[i] = neighbors.get(i, []) + [j]
    
    H = 10
    m = 1
    size = (5,5)

    # print the difference between the probability grid computed and the one from simulate_fires_and_compute_probability_grid
    cache = ProbaCache(p, q, neighbors, H)
    cache.compute_proba(H)
    theoretical_grid_dictionary = cache.proba
    
    # convert dictionary grid into numpy array with same shape as simulation
    theoretical_grid = np.zeros((H+1, size[0], size[1]))
    for key, value in theoretical_grid_dictionary.items():
        if len(key) == 2:  # only (i, t) keys
            i, t = key
            if t <= H:  # Only process time steps up to H
                theoretical_grid[t, i[0], i[1]] = value

    computed_grid = simulate_fires_and_compute_probability_grid(p, q, neighbors, H, m, size)
    
    # Convert theoretical grid to fire probabilities (1 - no_fire)
    theoretical_grid = 1 - theoretical_grid
    
    # Print difference grid for each time step
    for t in range(H+1):
        print(f"\nTime step {t}:")
        
        # Print theoretical grid in blue
        print_grid(theoretical_grid[t], f"Theoretical probabilities at t={t}", "\033[94m")
        
        # Print simulated grid in green
        print_grid(computed_grid[t], f"Simulated probabilities at t={t}", "\033[92m")
        
        # Print absolute difference grid in red
        difference_grid = np.abs(theoretical_grid[t] - computed_grid[t])
        print_grid(difference_grid, f"Absolute difference at t={t}", "\033[91m")
        print(f"Total absolute difference at t={t}: {np.sum(difference_grid):.3f}")
        
        # Calculate and print relative difference
        relative_diff = np.abs(theoretical_grid[t] - computed_grid[t]) / (theoretical_grid[t] + 1e-10)  # Add small epsilon to avoid division by zero
        print_grid(relative_diff, f"Relative difference at t={t}", "\033[93m")
        total_relative_diff = np.sum(difference_grid) / np.sum(theoretical_grid[t])
        print(f"Total relative difference at t={t}: {total_relative_diff:.3f} ({total_relative_diff*100:.1f}%)")

    #selected, value, cache = lazy_greedy(p, q, neighbors, H, m)
    # print("Selected:", selected)
    # print("E[T(x)]:", value)
    # cache.pretty_print(selected)
    # print("\n\n\n")
    # cache.pretty_print()




    # import numpy as np

    # # --- parameters -------------------------------------------------------------
    # grid_shape = (3, 3)
    # q = 0.9                           # spread probability to direct neighbour
    # p = np.zeros(grid_shape)          # spontaneous–infection probabilities
    # p[0, 0] = 0.05                     # only the top-left cell can start itself

    # T = 6                             # number of time-steps to simulate
    # # ---------------------------------------------------------------------------

    # # P[t, i, j]  =  probability that cell (i,j) is **healthy** at time t
    # P = np.ones((T + 1, *grid_shape))       # start with everybody healthy (t = 0)

    # def neighbours(i, j):
    #     """return list of direct-neighbour indices within the grid"""
    #     for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #         ni, nj = i + di, j + dj
    #         if 0 <= ni < grid_shape[0] and 0 <= nj < grid_shape[1]:
    #             yield ni, nj

    # for t in range(1, T + 1):
    #     for i in range(grid_shape[0]):
    #         for j in range(grid_shape[1]):
    #             prod = 1.0
    #             for ni, nj in neighbours(i, j):
    #                 prod *= 1 - q * (1 - P[t - 1, ni, nj])
    #             P[t, i, j] = P[t - 1, i, j] * (1 - p[i, j]) * prod

    # # infection probabilities after T steps
    # I_T = 1 - P[T]

    # # identify the cell with the highest infection probability
    # max_idx = np.unravel_index(np.argmax(I_T), grid_shape)
    # max_prob = I_T[max_idx]

    # print(f"Infection probabilities at t={T}:\n{I_T}\n")
    # print(f"Cell with highest infection probability at t={T}: {max_idx} with probability {max_prob:.4f}")
