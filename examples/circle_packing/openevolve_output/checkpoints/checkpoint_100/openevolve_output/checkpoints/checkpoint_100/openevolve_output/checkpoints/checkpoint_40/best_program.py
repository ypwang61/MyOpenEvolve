# EVOLVE-BLOCK-START
"""Advanced circle packing for n=26 circles using a specialized pattern and multi-stage optimization"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    that maximizes the sum of their radii using specialized patterns and optimization.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Try multiple specialized patterns known to be effective for n=26
    patterns = [
        initialize_pattern_greedy_corners,
        initialize_pattern_hexagonal_optimal,
        initialize_pattern_hybrid_optimal,
        initialize_pattern_variable_size,
        initialize_pattern_edge_optimized,
        initialize_pattern_research_based
    ]

    for pattern_func in patterns:
        # Initialize with a specialized pattern
        centers, radii = pattern_func(n)
        
        # Optimize with a two-stage approach
        centers, radii = optimize_packing(centers, radii)
        
        # Keep the best result
        sum_radii = np.sum(radii)
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, best_sum

def initialize_pattern_greedy_corners(n):
    """
    Initialize with a pattern that maximizes corner utilization with variable sizes.
    This pattern is based on mathematical research for optimal circle packing.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Four large corner circles - these are critical for optimal packing
    corner_radius = 0.1464
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius
    
    # Four circles at the midpoints of edges
    edge_radius = 0.1
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius
    
    # Center circle
    centers[8] = [0.5, 0.5]
    radii[8] = 0.12
    
    # Inner ring of 8 circles around center
    inner_radius = 0.095
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = radii[8] + inner_radius
        centers[9 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[9 + i] = inner_radius
    
    # Additional circles to fill gaps
    gap_radius = 0.08
    centers[17] = [0.25, 0.25]
    centers[18] = [0.75, 0.25]
    centers[19] = [0.25, 0.75]
    centers[20] = [0.75, 0.75]
    radii[17:21] = gap_radius
    
    # Smaller circles to fill remaining spaces
    small_radius = 0.07
    centers[21] = [0.3, 0.5]
    centers[22] = [0.7, 0.5]
    centers[23] = [0.5, 0.3]
    centers[24] = [0.5, 0.7]
    centers[25] = [0.5, 0.85]
    radii[21:26] = small_radius
    
    return centers, radii

def initialize_pattern_hexagonal_optimal(n):
    """
    Initialize with a hexagonal-based pattern with variable sizes.
    This is inspired by densest circle packing arrangements.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Large center circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.135
    
    # First hexagonal ring (6 circles)
    ring1_radius = 0.11
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + ring1_radius
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = ring1_radius
    
    # Second hexagonal ring (12 circles)
    ring2_radius = 0.095
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        dist = radii[0] + 2 * ring1_radius + ring2_radius
        centers[i + 7] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 7] = ring2_radius
    
    # Four corner circles - these are critical for optimal packing
    corner_radius = 0.1
    centers[19] = [corner_radius, corner_radius]
    centers[20] = [1 - corner_radius, corner_radius]
    centers[21] = [corner_radius, 1 - corner_radius]
    centers[22] = [1 - corner_radius, 1 - corner_radius]
    radii[19:23] = corner_radius
    
    # Fill remaining spaces with 3 smaller circles
    small_radius = 0.07
    centers[23] = [0.25, 0.5]
    centers[24] = [0.75, 0.5]
    centers[25] = [0.5, 0.25]
    radii[23:26] = small_radius
    
    return centers, radii

def initialize_pattern_hybrid_optimal(n):
    """
    Initialize with a hybrid pattern combining hexagonal and square grid elements.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Four large corner circles
    corner_radius = 0.1464
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius
    
    # Large center circle
    centers[4] = [0.5, 0.5]
    radii[4] = 0.125
    
    # Inner ring of 6 circles around center
    inner_radius = 0.105
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[4] + inner_radius
        centers[5 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[5 + i] = inner_radius
    
    # Midpoint edge circles
    edge_radius = 0.09
    centers[11] = [0.5, edge_radius]
    centers[12] = [0.5, 1 - edge_radius]
    centers[13] = [edge_radius, 0.5]
    centers[14] = [1 - edge_radius, 0.5]
    radii[11:15] = edge_radius
    
    # Additional circles to fill gaps
    gap_radius = 0.08
    centers[15] = [0.25, 0.25]
    centers[16] = [0.75, 0.25]
    centers[17] = [0.25, 0.75]
    centers[18] = [0.75, 0.75]
    radii[15:19] = gap_radius
    
    # Smaller circles for remaining spaces
    small_radius = 0.075
    positions = [
        [0.33, 0.5], [0.67, 0.5], [0.5, 0.33], [0.5, 0.67],
        [0.33, 0.33], [0.67, 0.33], [0.33, 0.67]
    ]
    
    for i in range(7):
        if i + 19 < n:
            centers[i + 19] = positions[i]
            radii[i + 19] = small_radius
    
    return centers, radii

def initialize_pattern_variable_size(n):
    """
    Initialize with a pattern specifically designed for n=26 with highly variable sizes.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Four large corner circles - mathematical optimum for corners
    corner_radius = 0.1464
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius
    
    # Four edge circles
    edge_radius = 0.11
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius
    
    # Large center circle
    centers[8] = [0.5, 0.5]
    radii[8] = 0.13
    
    # First ring of 6 medium circles
    medium_radius = 0.095
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[8] + medium_radius
        centers[9 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[9 + i] = medium_radius
    
    # Four circles near quarter positions
    quarter_radius = 0.08
    centers[15] = [0.25, 0.25]
    centers[16] = [0.75, 0.25]
    centers[17] = [0.25, 0.75]
    centers[18] = [0.75, 0.75]
    radii[15:19] = quarter_radius
    
    # Seven smaller circles to fill gaps
    small_radius = 0.07
    positions = [
        [0.33, 0.5], [0.67, 0.5], [0.5, 0.33], [0.5, 0.67],
        [0.33, 0.33], [0.67, 0.33], [0.33, 0.67]
    ]
    
    for i in range(7):
        if i + 19 < n:
            centers[i + 19] = positions[i]
            radii[i + 19] = small_radius
    
    return centers, radii

def initialize_pattern_edge_optimized(n):
    """
    Initialize with a pattern that maximizes edge utilization.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Four large corner circles
    corner_radius = 0.1464
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius
    
    # Eight edge circles (2 on each edge)
    edge_radius = 0.09
    edge_positions = [0.3, 0.7]
    
    # Bottom edge
    centers[4] = [edge_positions[0], edge_radius]
    centers[5] = [edge_positions[1], edge_radius]
    
    # Top edge
    centers[6] = [edge_positions[0], 1 - edge_radius]
    centers[7] = [edge_positions[1], 1 - edge_radius]
    
    # Left edge
    centers[8] = [edge_radius, edge_positions[0]]
    centers[9] = [edge_radius, edge_positions[1]]
    
    # Right edge
    centers[10] = [1 - edge_radius, edge_positions[0]]
    centers[11] = [1 - edge_radius, edge_positions[1]]
    
    radii[4:12] = edge_radius
    
    # Large center circle
    centers[12] = [0.5, 0.5]
    radii[12] = 0.13
    
    # Inner ring of 6 circles
    inner_radius = 0.095
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[12] + inner_radius
        centers[13 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[13 + i] = inner_radius
    
    # Fill remaining spaces with 7 smaller circles
    small_radius = 0.07
    positions = [
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.4, 0.4], [0.6, 0.4], [0.5, 0.75]
    ]
    
    for i in range(7):
        if i + 19 < n:
            centers[i + 19] = positions[i]
            radii[i + 19] = small_radius
    
    return centers, radii

def initialize_pattern_research_based(n):
    """
    Initialize with a pattern based on mathematical research for n=26.
    This uses the exact mathematical optimum for corner circles.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Four large corner circles - mathematical optimum is r = (3-2√2)/2 ≈ 0.1464
    corner_radius = (3 - 2 * np.sqrt(2)) / 2
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius
    
    # Edge circles
    edge_radius = 0.11
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius
    
    # Large center circle
    centers[8] = [0.5, 0.5]
    radii[8] = 0.135
    
    # First ring of 6 circles
    ring1_radius = 0.1
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[8] + ring1_radius
        centers[i + 9] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 9] = ring1_radius
    
    # Second partial ring of 4 circles
    ring2_radius = 0.085
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi / 4
        dist = radii[8] + 2 * ring1_radius + ring2_radius
        centers[i + 15] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 15] = ring2_radius
    
    # Fill remaining spaces with 7 smaller circles
    small_radius = 0.075
    positions = [
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.35, 0.5], [0.65, 0.5], [0.5, 0.35]
    ]
    
    for i in range(7):
        if i + 19 < n:
            centers[i + 19] = positions[i]
            radii[i + 19] = small_radius
    
    return centers, radii

def optimize_packing(centers, radii):
    """
    Optimize circle packing using a two-stage approach.
    First optimize positions with fixed radii, then optimize both.
    """
    n = len(centers)
    
    # Stage 1: Optimize positions with fixed radii
    def objective_positions(x):
        """Maximize minimum distance between circles (scaled by radii)"""
        current_centers = x.reshape((n, 2))
        min_dist_ratio = float('inf')
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                required_dist = radii[i] + radii[j]
                ratio = dist / required_dist if required_dist > 0 else float('inf')
                min_dist_ratio = min(min_dist_ratio, ratio)
                
        return -min_dist_ratio  # Maximize minimum distance ratio
    
    def constraint_positions(x):
        """Ensure circles stay within unit square and don't overlap"""
        current_centers = x.reshape((n, 2))
        constraints = []
        
        # Circles within unit square
        for i in range(n):
            constraints.append(current_centers[i, 0] - radii[i])  # x >= r
            constraints.append(1 - current_centers[i, 0] - radii[i])  # x <= 1-r
            constraints.append(current_centers[i, 1] - radii[i])  # y >= r
            constraints.append(1 - current_centers[i, 1] - radii[i])  # y <= 1-r
        
        # No overlapping
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                constraints.append(dist - radii[i] - radii[j])
                
        return np.array(constraints)
    
    # Optimize positions
    x0_positions = centers.flatten()
    cons_positions = {'type': 'ineq', 'fun': constraint_positions}
    
    res_positions = minimize(
        objective_positions, 
        x0_positions, 
        method='SLSQP', 
        constraints=cons_positions, 
        options={'maxiter': 300, 'ftol': 1e-6}
    )
    
    optimized_centers = res_positions.x.reshape((n, 2))
    
    # Stage 2: Optimize both positions and radii
    def objective_full(x):
        """Objective function to maximize sum of radii"""
        current_radii = x[2*n:]
        return -np.sum(current_radii)
    
    def constraint_full(x):
        """Constraints for full optimization"""
        current_centers = x[:2*n].reshape((n, 2))
        current_radii = x[2*n:]
        constraints = []
        
        # Circles within unit square
        for i in range(n):
            constraints.append(current_centers[i, 0] - current_radii[i])  # x >= r
            constraints.append(1 - current_centers[i, 0] - current_radii[i])  # x <= 1-r
            constraints.append(current_centers[i, 1] - current_radii[i])  # y >= r
            constraints.append(1 - current_centers[i, 1] - current_radii[i])  # y <= 1-r
            constraints.append(current_radii[i] - 0.01)  # Minimum radius
        
        # No overlapping
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                constraints.append(dist - current_radii[i] - current_radii[j] - 1e-6)
                
        return np.array(constraints)
    
    # Optimize positions and radii
    x0_full = np.concatenate([optimized_centers.flatten(), radii])
    cons_full = {'type': 'ineq', 'fun': constraint_full}
    
    res_full = minimize(
        objective_full, 
        x0_full, 
        method='SLSQP', 
        constraints=cons_full, 
        options={'maxiter': 500, 'ftol': 1e-8}
    )
    
    final_centers = res_full.x[:2*n].reshape((n, 2))
    final_radii = res_full.x[2*n:]
    
    # Final refinement with increased precision
    def refine_packing(centers, radii):
        """Refine the packing with a focused optimization"""
        x0 = np.concatenate([centers.flatten(), radii])
        
        res = minimize(
            objective_full, 
            x0, 
            method='SLSQP', 
            constraints=cons_full, 
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        refined_centers = res.x[:2*n].reshape((n, 2))
        refined_radii = res.x[2*n:]
        
        return refined_centers, refined_radii
    
    # Apply final refinement
    final_centers, final_radii = refine_packing(final_centers, final_radii)
    
    return final_centers, final_radii

# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    # plt.show()
    plt.savefig("circle_packing_s100.png")


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635
    # Uncomment to visualize:
    visualize(centers, radii)
# EVOLVE-BLOCK-END