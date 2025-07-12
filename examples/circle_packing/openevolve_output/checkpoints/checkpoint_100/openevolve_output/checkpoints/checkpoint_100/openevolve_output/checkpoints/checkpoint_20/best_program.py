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
        initialize_pattern_optimized_26_v3,  # More advanced initialization
        initialize_pattern_specialized_26_v2,
        initialize_pattern_hybrid_26_v2,
        initialize_pattern_billiard_26_v2,
        initialize_pattern_corner_optimized_26_v2,
        initialize_pattern_ring_26_v2
    ]

    for pattern_func in patterns:
        # Initialize with a specialized pattern
        centers, radii = pattern_func(n)

        # Combine and simplify optimization stages using scipy.optimize.minimize
        centers, radii = optimize_with_scipy(centers, radii)

        # Keep the best result
        sum_radii = np.sum(radii)
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, best_sum

def initialize_pattern_optimized_26_v3(n):
    """
    A highly optimized initialization pattern for 26 circles, focusing on efficient space filling.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Largest possible center circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.14  # Increased center radius

    # Ring of 6 circles around the center
    ring1_radius = 0.102
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + ring1_radius
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = ring1_radius

    # Second ring of 6 circles, offset from the first
    ring2_radius = 0.095
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        dist = radii[0] + 2 * ring1_radius + ring2_radius
        centers[i + 7] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 7] = ring2_radius

    # Place 4 circles near the corners
    corner_radius = 0.115
    centers[13] = [corner_radius, corner_radius]
    centers[14] = [1 - corner_radius, corner_radius]
    centers[15] = [corner_radius, 1 - corner_radius]
    centers[16] = [1 - corner_radius, 1 - corner_radius]
    radii[13:17] = corner_radius

    # Place 4 circles at the midpoints of the edges
    edge_radius = 0.09
    centers[17] = [0.5, edge_radius]
    centers[18] = [0.5, 1 - edge_radius]
    centers[19] = [edge_radius, 0.5]
    centers[20] = [1 - edge_radius, 0.5]
    radii[17:21] = edge_radius

    # Place 5 smaller circles to fill remaining gaps
    small_radius = 0.078
    centers[21] = [0.25, 0.25]
    centers[22] = [0.75, 0.25]
    centers[23] = [0.25, 0.75]
    centers[24] = [0.75, 0.75]
    centers[25] = [0.5, 0.85]  # Experiment with placement
    radii[21:26] = small_radius

    return centers, radii


def initialize_pattern_specialized_26_v2(n):
    """
    Initialize with a pattern specifically optimized for n=26 based on mathematical research.
    This pattern uses a hybrid approach with variable-sized circles.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Place 4 large circles in corners
    corner_radius = 0.118
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # Place 4 medium circles at midpoints of edges
    edge_radius = 0.103
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius

    # Place a large circle in the center
    centers[8] = [0.5, 0.5]
    radii[8] = 0.125

    # Place 8 circles in inner ring around center
    inner_radius = 0.098
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = radii[8] + inner_radius
        centers[9 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[9 + i] = inner_radius

    # Place 2 circles in outer ring
    outer_radius = 0.083
    for i in range(2):
        angle = 2 * np.pi * i / 2
        dist = radii[8] + 2 * inner_radius + outer_radius
        centers[17 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[17 + i] = outer_radius

    # Fill remaining spaces with 7 smaller circles
    small_radius = 0.073
    centers[19] = [0.25, 0.25]
    centers[20] = [0.75, 0.25]
    centers[21] = [0.25, 0.75]
    centers[22] = [0.75, 0.75]
    centers[23] = [0.35, 0.5]
    centers[24] = [0.65, 0.5]
    centers[25] = [0.5, 0.35]
    radii[19:26] = small_radius

    return centers, radii


def initialize_pattern_hybrid_26_v2(n):
    """
    Initialize with a hybrid pattern optimized for n=26 with strategic placement
    of variable-sized circles.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Center large circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.13

    # Inner ring (6 circles)
    inner_radius = 0.103
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + inner_radius
        centers[i + 1] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[i + 1] = inner_radius

    # Middle ring (8 circles, slightly offset)
    middle_radius = 0.093
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8
        dist = radii[0] + 2 * inner_radius + middle_radius
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
    edge_radius = 0.088
    centers[19] = [0.5, edge_radius]
    centers[20] = [0.5, 1 - edge_radius]
    centers[21] = [edge_radius, 0.5]
    centers[22] = [1 - edge_radius, 0.5]
    radii[19:23] = edge_radius

    # Fill remaining spaces with 3 smaller circles
    small_radius = 0.073
    centers[23] = [0.25, 0.25]
    centers[24] = [0.75, 0.25]
    centers[25] = [0.25, 0.75]
    radii[23:26] = small_radius

    return centers, radii


def initialize_pattern_billiard_26_v2(n):
    """
    Initialize with a billiard-table inspired pattern with circles along the edges
    and a triangular pattern in the center.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 4 corner circles
    corner_radius = 0.113
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # 8 edge circles (2 on each edge)
    edge_radius = 0.093
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

    # Triangular pattern in center (14 circles)
    inner_radius = 0.083

    # Center circle
    centers[12] = [0.5, 0.5]
    radii[12] = 0.105  # Slightly larger center circle

    # Inner hexagon (6 circles)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[12] + inner_radius
        centers[13 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[13 + i] = inner_radius

    # Outer partial ring (7 circles)
    outer_radius = 0.078
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi / 7
        dist = radii[12] + 2 * inner_radius + outer_radius
        centers[19 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[19 + i] = outer_radius

    return centers, radii


def initialize_pattern_corner_optimized_26_v2(n):
    """
    Initialize with a pattern that emphasizes optimal corner and edge utilization.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # 4 large corner circles
    corner_radius = 0.123
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius

    # 4 medium edge circles
    edge_radius = 0.098
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius

    # 4 smaller edge circles
    small_edge_radius = 0.083
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
    inner_radius = 0.078
    grid_positions = [0.3, 0.5, 0.7]

    count = 16
    for x in grid_positions:
        for y in grid_positions:
            if count < n:
                centers[count] = [x, y]
                # Larger circle in the center
                if x == 0.5 and y == 0.5:
                    radii[count] = 0.115
                else:
                    radii[count] = inner_radius
                count += 1

    # Fill any remaining slots
    while count < n:
        centers[count] = [0.4 + 0.2 * np.random.rand(), 0.4 + 0.2 * np.random.rand()]
        radii[count] = 0.073
        count += 1

    return centers, radii


def initialize_pattern_ring_26_v2(n):
    """
    Initialize with concentric rings of circles, optimized for n=26.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Center circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.133

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
    """
    n = len(centers)

    def objective(x):
        """Objective function to minimize (negative sum of radii)."""
        current_centers = x[:2 * n].reshape((n, 2))
        current_radii = x[2 * n:]
        return -np.sum(current_radii)

    def constraint(x):
        """Constraints: no overlapping circles, circles within the unit square."""
        current_centers = x[:2 * n].reshape((n, 2))
        current_radii = x[2 * n:]
        constraints = []

        # No overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(current_centers[i] - current_centers[j])
                constraints.append(dist - current_radii[i] - current_radii[j] - 1e-7)  # Add small tolerance

        # Circles within the unit square
        for i in range(n):
            constraints.append(current_centers[i, 0] - current_radii[i] + 1e-7)  # x >= r
            constraints.append(1 - current_centers[i, 0] - current_radii[i] + 1e-7)  # x <= 1 - r
            constraints.append(current_centers[i, 1] - current_radii[i] + 1e-7)  # y >= r
            constraints.append(1 - current_centers[i, 1] - current_radii[i] + 1e-7)  # y <= 1 - r
            constraints.append(current_radii[i] - 0.01) # radii >= 0.01

        return np.array(constraints)

    # Initial guess
    x0 = np.concatenate([centers.flatten(), radii])

    # Constraints
    cons = {'type': 'ineq', 'fun': constraint}

    # Optimization using SLSQP
    res = minimize(objective, x0, method='SLSQP', constraints=cons, options={'maxiter': 1000, 'ftol': 1e-7})

    # Extract optimized centers and radii
    optimized_centers = res.x[:2 * n].reshape((n, 2))
    optimized_radii = res.x[2 * n:]

    return optimized_centers, optimized_radii


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