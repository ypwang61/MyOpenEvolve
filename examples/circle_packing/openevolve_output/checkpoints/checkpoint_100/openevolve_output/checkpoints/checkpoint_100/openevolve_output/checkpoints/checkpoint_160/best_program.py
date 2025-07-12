# EVOLVE-BLOCK-START
"""Advanced circle packing for n=26 circles using specialized patterns and optimization techniques"""
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
        initialize_pattern_corner_optimized_26, # Keep this one as it's generally good
        initialize_pattern_research_26,
        initialize_pattern_hybrid_26,
        initialize_pattern_ring_26,
    ]

    for pattern_func in patterns:
        # Initialize with a specialized pattern
        centers, radii = pattern_func()
        
        # Ensure we have exactly 26 circles
        assert centers.shape[0] == n, f"Pattern function {pattern_func.__name__} returned {centers.shape[0]} circles instead of {n}"
        
        # Perform optimization
        try:
            centers, radii = optimize_with_scipy(centers, radii)
            
            # Keep the best result
            sum_radii = np.sum(radii)
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_centers = centers.copy()
                best_radii = radii.copy()
        except Exception as e:
            print(f"Optimization failed for {pattern_func.__name__}: {e}")
            continue

    return best_centers, best_radii, best_sum


def initialize_pattern_research_26():
    """
    Initialize with a pattern based on mathematical research for n=26,
    specifically targeting the known optimal arrangement.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Center circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.132
    
    # First ring - 6 circles in hexagonal arrangement
    ring1_radius = 0.107
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + ring1_radius + 0.002 # slight increase in distance
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = ring1_radius
    
    # Second ring - 12 circles
    ring2_radius = 0.097
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12  # Offset to stagger
        dist = radii[0] + 2 * ring1_radius + 0.015 # slight increase in distance
        centers[i + 7] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 7] = ring2_radius
    
    # Corner circles - 4 circles
    corner_radius = 0.115
    centers[19] = [corner_radius, corner_radius]
    centers[20] = [1 - corner_radius, corner_radius]
    centers[21] = [corner_radius, 1 - corner_radius]
    centers[22] = [1 - corner_radius, 1 - corner_radius]
    radii[19:23] = corner_radius
    
    # Edge circles - 3 circles
    edge_radius = 0.092
    centers[23] = [0.5, edge_radius]
    centers[24] = [edge_radius, 0.5]
    centers[25] = [1 - edge_radius, 0.5]
    radii[23:26] = edge_radius
    
    return centers, radii


def initialize_pattern_specialized_26():
    """
    Initialize with a pattern specifically optimized for n=26 based on mathematical research.
    This pattern uses a hybrid approach with variable-sized circles.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Place 4 large circles in corners
    corner_radius = 0.115
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # Place 4 medium circles at midpoints of edges
    edge_radius = 0.1
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius

    # Place a large circle in the center
    centers[8] = [0.5, 0.5]
    radii[8] = 0.12

    # Place 8 circles in inner ring around center
    inner_radius = 0.095
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = radii[8] + inner_radius
        centers[9 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[9 + i] = inner_radius

    # Place 9 circles in outer ring
    outer_radius = 0.08
    for i in range(9):
        angle = 2 * np.pi * i / 9
        dist = radii[8] + 2 * inner_radius + outer_radius
        centers[17 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[17 + i] = outer_radius

    return centers, radii


def initialize_pattern_hybrid_26():
    """
    Initialize with a hybrid pattern optimized for n=26 with strategic placement
    of variable-sized circles.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Center large circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.127

    # Inner ring (6 circles)
    inner_radius = 0.102
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + inner_radius + 0.002 # slight increase in distance
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = inner_radius

    # Middle ring (8 circles, slightly offset)
    middle_radius = 0.092
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8
        dist = radii[0] + 2 * inner_radius + middle_radius + 0.003 # slight increase in distance
        centers[i + 7] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 7] = middle_radius

    # Corner circles (4 larger circles)
    corner_radius = 0.113
    centers[15] = [corner_radius, corner_radius]
    centers[16] = [1 - corner_radius, corner_radius]
    centers[17] = [corner_radius, 1 - corner_radius]
    centers[18] = [1 - corner_radius, 1 - corner_radius]
    radii[15:19] = corner_radius

    # Edge circles (4 medium circles)
    edge_radius = 0.087
    centers[19] = [0.5, edge_radius]
    centers[20] = [0.5, 1 - edge_radius]
    centers[21] = [edge_radius, 0.5]
    centers[22] = [1 - edge_radius, 0.5]
    radii[19:23] = edge_radius

    # Fill remaining spaces with 3 smaller circles
    small_radius = 0.072
    centers[23] = [0.25, 0.25]
    centers[24] = [0.75, 0.25]
    centers[25] = [0.25, 0.75]
    radii[23:26] = small_radius

    return centers, radii



def initialize_pattern_corner_optimized_26():
    """
    Initialize with a pattern that emphasizes optimal corner and edge utilization.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 4 large corner circles
    corner_radius = 0.122
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # 4 medium edge circles
    edge_radius = 0.097
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius

    # 8 smaller edge circles
    small_edge_radius = 0.082
    centers[8] = [0.25, small_edge_radius]
    centers[9] = [0.75, small_edge_radius]
    centers[10] = [0.25, 1 - small_edge_radius]
    centers[11] = [0.75, 1 - small_edge_radius]
    centers[12] = [small_edge_radius, 0.25]
    centers[13] = [small_edge_radius, 0.75]
    centers[14] = [1 - small_edge_radius, 0.25]
    centers[15] = [1 - small_edge_radius, 0.75]
    radii[8:16] = small_edge_radius

    # Inner grid (10 circles)
    inner_radius = 0.077
    grid_positions = [0.3, 0.5, 0.7]
    
    count = 16
    grid_points = []
    for x in grid_positions:
        for y in grid_positions:
            grid_points.append((x, y))
    
    # Ensure we don't exceed 26 circles
    for x, y in grid_points[:n-count]:
        if x == 0.5 and y == 0.5:
            # Larger circle in the center
            centers[count] = [x, y]
            radii[count] = 0.115
        else:
            centers[count] = [x, y]
            radii[count] = inner_radius
        count += 1

    # Fill any remaining slots
    while count < n:
        centers[count] = [0.4 + 0.2 * (count - 16) / 10, 0.4 + 0.2 * ((count - 16) % 3) / 3]
        radii[count] = 0.07
        count += 1

    return centers, radii


def initialize_pattern_ring_26():
    """
    Initialize with concentric rings of circles, optimized for n=26.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Center circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.135

    # First ring (8 circles)
    ring1_radius = 0.098
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = radii[0] + ring1_radius
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = ring1_radius

    # Second ring (12 circles)
    ring2_radius = 0.088
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        dist = radii[0] + 2 * ring1_radius + ring2_radius
        centers[i + 9] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 9] = ring2_radius

    # Corner circles (4 circles)
    corner_radius = 0.093
    centers[21] = [corner_radius, corner_radius]
    centers[22] = [1 - corner_radius, corner_radius]
    centers[23] = [corner_radius, 1 - corner_radius]
    centers[24] = [1 - corner_radius, 1 - corner_radius]
    radii[21:25] = corner_radius

    # One extra circle
    centers[25] = [0.5, 0.15]
    radii[25] = 0.083

    return centers, radii



def optimize_with_scipy(centers, radii):
    """
    Optimize circle positions and radii using scipy.optimize.minimize.
    Uses a multi-stage approach with progressive refinement to avoid local minima.
    """
    n = len(centers)
    
    # Stage 1: Optimize positions with fixed radii
    def objective_positions(x):
        """Objective function for position optimization."""
        current_centers = x.reshape((n, 2))
        return calculate_penalty(current_centers, radii)
    
    x0_positions = centers.flatten()
    bounds_positions = [(0, 1) for _ in range(2*n)]
    
    res_positions = minimize(
        objective_positions, 
        x0_positions, 
        method='L-BFGS-B',
        bounds=bounds_positions,
        options={'maxiter': 250, 'ftol': 1e-7} # Reduced iterations slightly
    )
    
    improved_centers = res_positions.x.reshape((n, 2))
    
    # Stage 2: Optimize radii with fixed positions
    def objective_radii(r):
        """Objective function for radii optimization (negative sum of radii)."""
        return -np.sum(r)
    
    def constraint_radii(r):
        """Constraints for radii optimization."""
        constraints = []
        
        # No overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(improved_centers[i] - improved_centers[j])
                constraints.append(dist - r[i] - r[j])
        
        # Circles within the unit square
        for i in range(n):
            constraints.append(improved_centers[i, 0] - r[i])  # x >= r
            constraints.append(1 - improved_centers[i, 0] - r[i])  # x <= 1 - r
            constraints.append(improved_centers[i, 1] - r[i])  # y >= r
            constraints.append(1 - improved_centers[i, 1] - r[i])  # y <= 1 - r
        
        return np.array(constraints)
    
    cons_radii = {'type': 'ineq', 'fun': constraint_radii}
    
    res_radii = minimize(
        objective_radii, 
        radii, 
        method='SLSQP',
        constraints=cons_radii,
        bounds=[(0.01, 0.5) for _ in range(n)],
        options={'maxiter': 350, 'ftol': 1e-7} # Reduced iterations slightly
    )
    
    improved_radii = res_radii.x
    
    # Stage 3: Joint optimization with progressive refinement
    best_centers = improved_centers.copy()
    best_radii = improved_radii.copy()
    best_sum = np.sum(best_radii)
    
    def objective_joint(x):
        """Objective function to minimize (negative sum of radii)."""
        current_centers = x[:2*n].reshape((n, 2))
        current_radii = x[2*n:]
        return -np.sum(current_radii)
    
    def constraint_joint(x):
        """Constraints: no overlapping circles, circles within the unit square."""
        current_centers = x[:2*n].reshape((n, 2))
        current_radii = x[2*n:]
        constraints = []
        
        # No overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                constraints.append(dist - current_radii[i] - current_radii[j] - 1e-6) # Add a small tolerance
        
        # Circles within the unit square
        for i in range(n):
            constraints.append(current_centers[i, 0] - current_radii[i])  # x >= r
            constraints.append(1 - current_centers[i, 0] - current_radii[i])  # x <= 1 - r
            constraints.append(current_centers[i, 1] - current_radii[i])  # y >= r
            constraints.append(1 - current_centers[i, 1] - current_radii[i])  # y <= 1 - r
        
        return np.array(constraints)
    
    x0_joint = np.concatenate([best_centers.flatten(), best_radii])
    bounds_joint = [(0, 1) for _ in range(2*n)] + [(0.01, 0.5) for _ in range(n)]
    cons_joint = {'type': 'ineq', 'fun': constraint_joint}
    
    res_joint = minimize(
        objective_joint, 
        x0_joint, 
        method='SLSQP',
        constraints=cons_joint,
        bounds=bounds_joint,
        options={'maxiter': 500, 'ftol': 1e-8} # Reduced iterations and tolerance
    )
    
    refined_centers = res_joint.x[:2*n].reshape((n, 2))
    refined_radii = res_joint.x[2*n:]
    refined_sum = np.sum(refined_radii)
    
    if refined_sum > best_sum:
        return refined_centers, refined_radii
    else:
        return best_centers, best_radii


def calculate_penalty(centers, radii):
    """
    Calculate penalty for overlapping circles or circles outside the unit square.
    Used for optimization.
    """
    n = len(centers)
    penalty = 0.0
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap**2 * 150  # Increased penalty weight
    
    # Penalty for circles outside the unit square
    for i in range(n):
        # Left boundary
        if centers[i, 0] - radii[i] < 0:
            penalty += (radii[i] - centers[i, 0])**2 * 150
        
        # Right boundary
        if centers[i, 0] + radii[i] > 1:
            penalty += (centers[i, 0] + radii[i] - 1)**2 * 150
        
        # Bottom boundary
        if centers[i, 1] - radii[i] < 0:
            penalty += (radii[i] - centers[i, 1])**2 * 150
        
        # Top boundary
        if centers[i, 1] + radii[i] > 1:
            penalty += (centers[i, 1] + radii[i] - 1)**2 * 150
    
    return penalty


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
    ax.set_aspect('equal')

    # Draw circles
    for i in range(len(centers)):
        circle = Circle(centers[i], radii[i], edgecolor='black', facecolor='skyblue', alpha=0.7)
        ax.add_patch(circle)

    # Show plot
    plt.title('Circle Packing in Unit Square (n=26)')
    # plt.show()

    plt.savefig('circle_packing_26_160.png')
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    visualize(centers, radii)