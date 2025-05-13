import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter

def rbf(X, X_, ls=1):
    """Radial basis function kernel"""
    X = X.reshape((-1,1)) if len(X.shape) == 1 else X
    X_ = X_.reshape((-1,1)) if len(X_.shape) == 1 else X_
    return np.exp(-(1/(2*ls)) * np.power(distance_matrix(X, X_), 2))

def fit_lgcp(cell_counts, grid_coords, ls_prior=2.0, variance_prior=10.0, threshold=0):
    """
    Fit a Log-Gaussian Cox Process to observed cell counts, focusing on nonzero cells.
    
    Args:
        cell_counts: Array of observed counts in each cell
        grid_coords: Array of coordinates for each cell (N x 2)
        ls_prior: Prior for length scale parameter
        variance_prior: Prior for variance parameter
        threshold: Count threshold for including cells (default: 0)
        
    Returns:
        trace: PyMC trace object containing posterior samples
    """
    # Find cells with counts above threshold
    nonzero_mask = cell_counts > threshold
    nonzero_counts = cell_counts[nonzero_mask]
    nonzero_coords = grid_coords[nonzero_mask]
    
    print(f"Modeling {len(nonzero_coords)} cells out of {len(grid_coords)} total cells")
    print(f"Reduction factor: {len(grid_coords)/len(nonzero_coords):.1f}x")
    
    N = len(nonzero_counts)
    
    with pm.Model() as lgcp_model:
        # Priors for GP parameters
        rho = pm.HalfNormal("rho", sigma=ls_prior, initval=1.0)  # length scale
        variance = pm.HalfNormal("variance", sigma=variance_prior, initval=1.0)  # variance
        mu = pm.Normal("mu", 0.0, 3.0, initval=0.0)  # mean
        
        # Define the GP
        cov_func = variance * pm.gp.cov.ExpQuad(2, ls=rho)  # RBF kernel in 2D
        mean_func = pm.gp.mean.Constant(mu)
        gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
        
        # GP prior (only for nonzero cells)
        log_intensity = gp.prior("log_intensity", X=nonzero_coords)
        
        # Convert to intensity
        intensity = pm.math.exp(log_intensity)
        
        # Area of each cell (assuming uniform grid)
        area_per_cell = 1.0  # adjust if needed
        
        # Expected counts
        rates = intensity * area_per_cell
        
        # Likelihood (only for nonzero cells)
        counts = pm.Poisson("counts", mu=rates, observed=nonzero_counts)
        
        # Fit the model
        approx = pm.fit(n=5000, method='advi', 
                       obj_optimizer=pm.adam(learning_rate=0.01),
                       callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
        
        trace = approx.sample(1000)
        
    return trace, nonzero_mask

def predict_full_grid(trace, grid_coords, nonzero_mask, jitter=1e-6):
    """
    Predict intensity for the full grid using the fitted model.
    
    Args:
        trace: PyMC trace object from fit_lgcp
        grid_coords: Full grid coordinates
        nonzero_mask: Boolean mask indicating which cells were used in fitting
        jitter: Jitter to add to the kernel matrix for numerical stability
        
    Returns:
        full_intensity: Predicted intensity for all cells
    """
    # Get the GP parameters from the trace
    rho = trace.posterior['rho'].mean().item()
    variance = trace.posterior['variance'].mean().item()
    mu = trace.posterior['mu'].mean().item()
    
    X_nonzero = grid_coords[nonzero_mask]
    X_pred = grid_coords
    K_nn = rbf(X_nonzero, X_nonzero, ls=rho) * variance
    K_star_n = rbf(X_pred, X_nonzero, ls=rho) * variance
    K_nn += jitter * np.eye(K_nn.shape[0])
    nonzero_log_intensity = trace.posterior['log_intensity'].mean(dim=['chain', 'draw']).values
    full_log_intensity = mu + K_star_n @ np.linalg.solve(K_nn, nonzero_log_intensity - mu)
    full_intensity = np.exp(full_log_intensity)
    
    return full_intensity

def plot_posterior_predictions(trace, grid_coords, cell_counts, true_intensity=None, nonzero_mask=None):
    """
    Plot posterior predictions and compare with observed data.
    
    Args:
        trace: PyMC trace object
        grid_coords: Array of coordinates for each cell
        cell_counts: Observed counts
        true_intensity: Optional true intensity for comparison
        nonzero_mask: Optional mask indicating which cells were used in fitting
    """
    if nonzero_mask is not None:
        # Use the full grid prediction
        post_pred = predict_full_grid(trace, grid_coords, nonzero_mask)
    else:
        # Get posterior mean predictions
        post_pred = np.exp(trace.posterior['log_intensity'].mean(dim=['chain', 'draw']).values)
    
    # Reshape for plotting
    grid_size = int(np.sqrt(len(grid_coords)))
    post_pred_grid = post_pred.reshape(grid_size, grid_size)
    counts_grid = cell_counts.reshape(grid_size, grid_size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot posterior mean intensity
    im1 = ax1.imshow(post_pred_grid, cmap='viridis')
    ax1.set_title('Posterior Mean Intensity')
    plt.colorbar(im1, ax=ax1)
    
    # Plot observed counts
    im2 = ax2.imshow(counts_grid, cmap='viridis')
    ax2.set_title('Observed Counts')
    plt.colorbar(im2, ax=ax2)
    
    if true_intensity is not None:
        true_intensity_grid = true_intensity.reshape(grid_size, grid_size)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(true_intensity_grid, cmap='viridis')
        ax.set_title('True Intensity')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

def sample_from_posterior(trace, grid_coords, n_samples=1, area=None):
    """
    Sample points from the fitted LGCP posterior.
    
    Args:
        trace: PyMC trace object from fit_lgcp
        grid_coords: Array of coordinates for each cell
        n_samples: Number of samples to generate
        area: Optional tuple (x_min, x_max, y_min, y_max) defining sampling area.
              If None, uses the same area as grid_coords.
    
    Returns:
        List of arrays, each containing sampled points (x, y) for each sample
    """
    # Get posterior samples of the intensity
    post_samples = np.exp(trace.posterior['log_intensity'].values)
    n_chains, n_draws = post_samples.shape[:2]
    
    # Determine sampling area
    if area is None:
        x_min, y_min = grid_coords.min(axis=0)
        x_max, y_max = grid_coords.max(axis=0)
    else:
        x_min, x_max, y_min, y_max = area
    
    # Generate samples
    all_samples = []
    for _ in range(n_samples):
        # Randomly select a chain and draw
        chain_idx = np.random.randint(0, n_chains)
        draw_idx = np.random.randint(0, n_draws)
        intensity = post_samples[chain_idx, draw_idx]
        
        # Normalize intensity to get probability distribution
        intensity = intensity / intensity.sum()
        
        # Sample points using rejection sampling
        n_points = np.random.poisson(intensity.sum())
        sampled_points = []
        
        for _ in range(n_points):
            # Sample a cell according to intensity
            cell_idx = np.random.choice(len(grid_coords), p=intensity)
            cell_center = grid_coords[cell_idx]
            
            # Add small random offset within cell
            cell_size = (x_max - x_min) / (np.sqrt(len(grid_coords)) - 1)
            offset = np.random.uniform(-cell_size/2, cell_size/2, size=2)
            point = cell_center + offset
            
            # Ensure point is within bounds
            point[0] = np.clip(point[0], x_min, x_max)
            point[1] = np.clip(point[1], y_min, y_max)
            
            sampled_points.append(point)
        
        # Convert list of points to numpy array with shape (n_points, 2)
        all_samples.append(np.array(sampled_points).reshape(-1, 2))
    
    return all_samples

def plot_samples(samples, grid_coords, true_intensity=None, n_cols=2):
    """
    Plot sampled points from the posterior.
    
    Args:
        samples: List of arrays containing sampled points
        grid_coords: Array of coordinates for each cell
        true_intensity: Optional true intensity for comparison
        n_cols: Number of columns in the plot grid
    """
    n_samples = len(samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Determine plot limits
    x_min, y_min = grid_coords.min(axis=0)
    x_max, y_max = grid_coords.max(axis=0)
    
    for i, (ax, points) in enumerate(zip(axes, samples)):
        if len(points) > 0:  # Only plot if we have points
            ax.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.5, s=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'Sample {i+1} (n={len(points)})')
    
    # Hide unused subplots
    for ax in axes[len(samples):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Harder dataset parameters
    grid_size = 200  # much larger grid
    mean_log_intensity = -4.5
    variance = 0.5
    smoothness = 10  # Higher = smoother (sigma for gaussian_filter)

    # Create grid coordinates
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_coords = np.column_stack((X.ravel(), Y.ravel()))

    # FAST: Generate true log-intensity using a Gaussian filter on white noise
    noise = np.random.randn(grid_size, grid_size)
    true_log_intensity = mean_log_intensity + np.sqrt(variance) * gaussian_filter(noise, sigma=smoothness)
    true_log_intensity = true_log_intensity.ravel()
    true_intensity = np.exp(true_log_intensity)

    # Generate observed counts (sparse)
    cell_counts = np.random.poisson(true_intensity)

    print(f"Total number of points: {cell_counts.sum()}")
    print(f"Fraction of empty cells: {(cell_counts == 0).mean():.2%}")

    # Test with threshold=0
    trace, nonzero_mask = fit_lgcp(cell_counts, grid_coords, threshold=0)

    # Test with threshold=1
    # trace_1, nonzero_mask_1 = fit_lgcp(cell_counts, grid_coords, threshold=1)
    
    # Plot results
    plot_posterior_predictions(trace, grid_coords, cell_counts, true_intensity, nonzero_mask)
    # plot_posterior_predictions(trace_1, grid_coords, cell_counts, true_intensity, nonzero_mask_1)
    
    # Print parameter estimates
    print("\nParameter Estimates:")
    print(f"Length Scale: {trace.posterior['rho'].mean().item():.3f}")
    print(f"Variance: {trace.posterior['variance'].mean().item():.3f}")
    print(f"Mean: {trace.posterior['mu'].mean().item():.3f}")
    
    # Sample from posterior
    print("\nSampling from posterior...")
    samples = sample_from_posterior(trace, grid_coords, n_samples=4)
    
    # Plot samples
    plot_samples(samples, grid_coords)
    
    # Print sample statistics
    print("\nSample Statistics:")
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {len(sample)} points")
    
    
    