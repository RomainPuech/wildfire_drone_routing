import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter
import pandas as pd

# --- Load wildfire data (replace with your actual data loading logic) ---
# Example: data = pd.read_csv('wildfire_data.csv')
# Columns: x, y, marks (optional)
# For demonstration, we assume 'data' is already loaded as a DataFrame

# --- Rectangle corners and grid setup (replace with your actual values) ---
# x_s, y_s: coordinates of the 4 corners of the rectangle (in order)
# layout_dict, layout, etc. should be defined as in your notebook
# For demonstration, we assume these are already defined

# --- Parameters ---
cells_x = layout_dict[layout]['width']
cells_y = layout_dict[layout]['height']

P0 = np.array([x_s[0], y_s[0]])  # Reference corner
P1 = np.array([x_s[1], y_s[1]])  # Adjacent corner (defines one edge)
P3 = np.array([x_s[3], y_s[3]])  # Other adjacent corner

vec_u = (P1 - P0) / cells_x  # Step along one edge
vec_v = (P3 - P0) / cells_y  # Step along adjacent edge

# --- Precompute inverse lattice matrix ---
M = np.column_stack((vec_u, vec_v))  # 2x2 matrix: [vec_u | vec_v]
M_inv = np.linalg.inv(M)  # Inverse matrix

# --- Initialize counts ---
cell_counts = np.zeros((cells_x, cells_y), dtype=int)

# --- Process all points efficiently ---
points = data[['x', 'y']].to_numpy()  # shape (N, 2)
relative_points = points - P0  # shift to origin
lattice_coords = relative_points @ M_inv.T  # shape (N, 2)
i_indices = np.floor(lattice_coords[:, 0]).astype(int)
j_indices = np.floor(lattice_coords[:, 1]).astype(int)
valid_mask = (i_indices >= 0) & (i_indices < cells_x) & (j_indices >= 0) & (j_indices < cells_y)
valid_i = i_indices[valid_mask]
valid_j = j_indices[valid_mask]
for i, j in zip(valid_i, valid_j):
    cell_counts[i, j] += 1

# --- Compute centroids of the cells ---
centroids = np.array([
    [P0 + i * vec_u + j * vec_v for j in range(cells_y)]
    for i in range(cells_x)
])
centroids_flat = centroids.reshape(-1, 2)
cell_counts_flat = cell_counts.flatten()

# --- LGCP Model ---
def rbf(X, X_, ls=1):
    X = X.reshape((-1,1)) if len(X.shape) == 1 else X
    X_ = X_.reshape((-1,1)) if len(X_.shape) == 1 else X_
    return np.exp(-(1/(2*ls)) * np.power(distance_matrix(X, X_), 2))

def fit_lgcp(cell_counts, grid_coords, ls_prior=2.0, variance_prior=10.0, threshold=0):
    nonzero_mask = cell_counts > threshold
    nonzero_counts = cell_counts[nonzero_mask]
    nonzero_coords = grid_coords[nonzero_mask]
    print(f"Modeling {len(nonzero_coords)} cells out of {len(grid_coords)} total cells")
    print(f"Reduction factor: {len(grid_coords)/len(nonzero_coords):.1f}x")
    with pm.Model() as lgcp_model:
        rho = pm.HalfNormal("rho", sigma=ls_prior, initval=1.0)
        variance = pm.HalfNormal("variance", sigma=variance_prior, initval=1.0)
        mu = pm.Normal("mu", 0.0, 3.0, initval=0.0)
        cov_func = variance * pm.gp.cov.ExpQuad(2, ls=rho)
        mean_func = pm.gp.mean.Constant(mu)
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        log_intensity = gp.prior("log_intensity", X=nonzero_coords)
        intensity = pm.math.exp(log_intensity)
        area_per_cell = 1.0
        rates = intensity * area_per_cell
        counts = pm.Poisson("counts", mu=rates, observed=nonzero_counts)
        approx = pm.fit(n=5000, method='advi', 
                       obj_optimizer=pm.adam(learning_rate=0.01),
                       callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
        trace = approx.sample(1000)
    return trace, nonzero_mask

def predict_full_grid(trace, grid_coords, nonzero_mask, jitter=1e-6):
    rho = trace.posterior['rho'].mean().item()
    variance = trace.posterior['variance'].mean().item()
    mu = trace.posterior['mu'].mean().item()
    K = rbf(grid_coords, grid_coords, ls=rho) * variance
    nonzero_log_intensity = trace.posterior['log_intensity'].mean(dim=['chain', 'draw']).values
    K_nn = K[nonzero_mask][:, nonzero_mask]
    K_nm = K[nonzero_mask]
    K_nn += jitter * np.eye(K_nn.shape[0])
    full_log_intensity = mu + K_nm.T @ np.linalg.solve(K_nn, nonzero_log_intensity - mu)
    full_intensity = np.exp(full_log_intensity)
    return full_intensity

def plot_posterior_predictions(trace, grid_coords, cell_counts, true_intensity=None, nonzero_mask=None):
    if nonzero_mask is not None:
        post_pred = predict_full_grid(trace, grid_coords, nonzero_mask)
    else:
        post_pred = np.exp(trace.posterior['log_intensity'].mean(dim=['chain', 'draw']).values)
    grid_size_x = cells_x
    grid_size_y = cells_y
    post_pred_grid = post_pred.reshape(grid_size_x, grid_size_y)
    counts_grid = cell_counts.reshape(grid_size_x, grid_size_y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(post_pred_grid, cmap='viridis')
    ax1.set_title('Posterior Mean Intensity')
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(counts_grid, cmap='viridis')
    ax2.set_title('Observed Counts')
    plt.colorbar(im2, ax=ax2)
    if true_intensity is not None:
        true_intensity_grid = true_intensity.reshape(grid_size_x, grid_size_y)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(true_intensity_grid, cmap='viridis')
        ax.set_title('True Intensity')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

# --- Run LGCP on wildfire data ---
if __name__ == "__main__":
    print(f"Total number of points: {cell_counts_flat.sum()}")
    print(f"Fraction of empty cells: {(cell_counts_flat == 0).mean():.2%}")
    trace, nonzero_mask = fit_lgcp(cell_counts_flat, centroids_flat, threshold=1)
    plot_posterior_predictions(trace, centroids_flat, cell_counts_flat, nonzero_mask=nonzero_mask)
    print("\nParameter Estimates:")
    print(f"Length Scale: {trace.posterior['rho'].mean().item():.3f}")
    print(f"Variance: {trace.posterior['variance'].mean().item():.3f}")
    print(f"Mean: {trace.posterior['mu'].mean().item():.3f}") 