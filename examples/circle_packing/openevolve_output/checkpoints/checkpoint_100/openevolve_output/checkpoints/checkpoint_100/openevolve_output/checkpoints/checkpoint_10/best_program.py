# EVOLVE-BLOCK-START
"""Advanced circle packing for n=26 circles using a specialized pattern and multi-stage optimization"""
import numpy as np


def construct_packing():
    """
    Construct an optimized arrangement of 26 circles in a unit square
    that maximizes the sum of their radii using specialized patterns.

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

    # Try multiple initializations and optimization strategies
    for strategy in range(8):  # Increased strategies
        # Initialize with different patterns
        if strategy == 0:
            centers, radii = initialize_pattern_hexagonal(n)
        elif strategy == 1:
            centers, radii = initialize_pattern_grid(n)
        elif strategy == 2:
            centers, radii = initialize_pattern_hybrid(n)
        elif strategy == 3:
            centers, radii = initialize_pattern_corner_biased(n)  # New strategy
        elif strategy == 4:
            centers, radii = initialize_pattern_random(n)  # New strategy
        elif strategy == 5:
            centers, radii = initialize_pattern_triangular(n)
        elif strategy == 6:
             centers, radii = initialize_pattern_adaptive_grid(n)
        else:
            centers, radii = initialize_pattern_radial(n)


        # Multi-stage optimization
        centers, radii = optimize_stage1(centers, radii, iterations=300)
        centers, radii = optimize_stage2(centers, radii, iterations=300)
        centers, radii = optimize_stage3(centers, radii, iterations=500)  # Increased iterations

        # Keep the best result
        sum_radii = np.sum(radii)
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, best_sum


def initialize_pattern_hexagonal(n):
    """Initialize with a hexagonal pattern with variable sizes"""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Center circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.11  # Larger center circle

    # First hexagonal ring (6 circles)
    count = 1
    ring_radius = 0.095
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + ring_radius
        centers[count] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[count] = ring_radius
        count += 1

    # Second hexagonal ring (12 circles)
    ring_radius = 0.085
    for i in range(12):
        angle = 2 * np.pi * i / 12
        dist = radii[0] + 2 * 0.095 + ring_radius
        centers[count] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[count] = ring_radius
        count += 1

    # Corner circles (4 circles)
    corner_radius = 0.09
    centers[count] = [0.1, 0.1]
    radii[count] = corner_radius
    count += 1
    centers[count] = [0.9, 0.1]
    radii[count] = corner_radius
    count += 1
    centers[count] = [0.1, 0.9]
    radii[count] = corner_radius
    count += 1
    centers[count] = [0.9, 0.9]
    radii[count] = corner_radius
    count += 1

    # Edge circles (3 circles)
    edge_radius = 0.07
    centers[count] = [0.5, 0.05]
    radii[count] = edge_radius
    count += 1
    centers[count] = [0.5, 0.95]
    radii[count] = edge_radius
    count += 1
    centers[count] = [0.05, 0.5]
    radii[count] = edge_radius

    return centers, radii


def initialize_pattern_grid(n):
    """Initialize with a grid pattern with variable sizes"""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Create a 5x5 grid (25 circles) plus 1 in center
    grid_size = 5
    spacing = 1.0 / grid_size
    count = 0

    # Grid circles
    for i in range(grid_size):
        for j in range(grid_size):
            if count < 25:
                centers[count] = [spacing / 2 + i * spacing, spacing / 2 + j * spacing]

                # Larger circles in the middle, smaller at edges
                dist_to_center = np.sqrt((centers[count][0] - 0.5) ** 2 + (centers[count][1] - 0.5) ** 2)
                radii[count] = 0.07 + 0.04 * (1 - dist_to_center**2) # Quadratic falloff
                count += 1

    # One extra circle in the center with variable size
    centers[count] = [0.5, 0.5]
    radii[count] = 0.07

    return centers, radii


def initialize_pattern_hybrid(n):
    """Initialize with a hybrid pattern optimized for n=26"""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Based on known good patterns for n=26
    # Central large circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.12

    # Inner ring (6 circles)
    count = 1
    inner_radius = 0.1
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + inner_radius
        centers[count] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[count] = inner_radius
        count += 1

    # Middle ring (6 circles)
    middle_radius = 0.09
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        dist = radii[0] + 2 * inner_radius + middle_radius
        centers[count] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[count] = middle_radius
        count += 1

    # Outer partial ring (9 circles)
    outer_radius = 0.08
    for i in range(9):
        angle = 2 * np.pi * i / 9
        dist = radii[0] + 2 * inner_radius + 2 * middle_radius
        centers[count] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[count] = outer_radius
        count += 1

    # Corner circles (4 circles)
    corner_radius = 0.06
    centers[count] = [corner_radius, corner_radius]
    radii[count] = corner_radius
    count += 1
    centers[count] = [1 - corner_radius, corner_radius]
    radii[count] = corner_radius
    count += 1
    centers[count] = [corner_radius, 1 - corner_radius]
    radii[count] = corner_radius
    count += 1
    centers[count] = [1 - corner_radius, 1 - corner_radius]
    radii[count] = corner_radius

    return centers, radii


def initialize_pattern_corner_biased(n):
    """Initialize with larger circles in the corners"""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    corner_radius = 0.10
    centers[0] = [corner_radius, corner_radius]
    radii[0] = corner_radius
    centers[1] = [1 - corner_radius, corner_radius]
    radii[1] = corner_radius
    centers[2] = [corner_radius, 1 - corner_radius]
    radii[2] = corner_radius
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[3] = corner_radius

    remaining = n - 4
    grid_size = int(np.sqrt(remaining))
    spacing = (1 - 2 * corner_radius) / (grid_size - 1) if grid_size > 1 else 0.5

    count = 4
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n:
                x = corner_radius + i * spacing
                y = corner_radius + j * spacing
                centers[count] = [x, y]
                radii[count] = 0.06 + 0.02 * np.random.rand()
                count += 1

    return centers, radii

def initialize_pattern_random(n):
    """Initialize with random positions and radii"""
    centers = np.random.rand(n, 2)
    radii = np.random.rand(n) * 0.06 + 0.05  # Radii between 0.05 and 0.11
    return centers, radii

def initialize_pattern_triangular(n):
    """Initialize with a triangular lattice pattern, denser packing."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    radius = 0.075  # Initial radius, will be adjusted.
    x_start = radius
    y_start = radius
    x_spacing = 2 * radius
    y_spacing = np.sqrt(3) * radius
    count = 0

    x = x_start
    y = y_start

    while count < n:
        centers[count] = [x, y]
        radii[count] = radius
        count += 1

        x += x_spacing
        if x > 1 - radius:
            x = x_start + (x_spacing / 2 if int(y / y_spacing) % 2 == 0 else 0) # Stagger rows
            y += y_spacing
            if y > 1 - radius:
                radius *= 0.9 # Reduce radius slightly if too many are needed.
                x_start = radius
                y_start = radius
                x_spacing = 2 * radius
                y_spacing = np.sqrt(3) * radius
                x = x_start
                y = y_start
                count = 0 #Restart packing with smaller circles

    return centers[:n], radii[:n]

def initialize_pattern_adaptive_grid(n):
    """Initialize with an adaptive grid pattern, denser in the center."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    count = 0

    # Place a larger circle in the center
    centers[0] = [0.5, 0.5]
    radii[0] = 0.12
    count += 1

    # Adaptively fill the rest of the space with smaller circles
    remaining = n - 1
    grid_density = 4  # Adjust this for density
    while count < n:
        for i in range(grid_density):
            for j in range(grid_density):
                if count < n:
                    x = (i + 0.5) / grid_density
                    y = (j + 0.5) / grid_density
                    dist_to_center = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
                    if dist_to_center > 0.15: #Avoid overlapping center circle
                        centers[count] = [x, y]
                        radii[count] = 0.06 + 0.02 * np.random.rand()
                        count += 1
        grid_density += 1 #Increase density if we haven't placed all circles

    return centers, radii

def initialize_pattern_radial(n):
    """Initialize with a radial pattern from the center."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    count = 0

    centers[0] = [0.5, 0.5]
    radii[0] = 0.12
    count += 1

    num_rings = 4
    circles_per_ring = [6, 8, 8, 3]  # Adjust distribution for different ring densities
    ring_radii = [0.10, 0.09, 0.08, 0.07] #Radius distribution

    for i in range(num_rings):
        num_circles = circles_per_ring[i]
        ring_radius = ring_radii[i]
        for j in range(num_circles):
            if count < n:
                angle = 2 * np.pi * j / num_circles
                dist = radii[0] + (i + 1) * 0.10
                centers[count] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
                radii[count] = ring_radius
                count += 1

    return centers, radii


def optimize_stage1(centers, radii, iterations):
    """
    First optimization stage: Resolve overlaps and establish basic structure
    """
    n = len(centers)
    velocity = np.zeros_like(centers)
    repulsion_strength = 25.0  # Increased
    wall_repulsion = 25.0  # Increased
    dt = 0.01
    dampening = 0.95
    gravity = 0.001  # Add a weak gravity towards the center, reduced

    for iter in range(iterations):
        forces = np.zeros_like(centers)
        temperature = 1.0 - iter / iterations  # Simulated annealing temperature

        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    overlap = min_dist - dist
                    direction = dist_vec / (dist + 1e-8)
                    force = repulsion_strength * overlap * direction
                    forces[i] += force
                    forces[j] -= force

        # Wall repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]

            if x < r:
                forces[i, 0] += wall_repulsion * (r - x)
            if x > 1 - r:
                forces[i, 0] -= wall_repulsion * (x - (1 - r))
            if y < r:
                forces[i, 1] += wall_repulsion * (r - y)
            if y > 1 - r:
                forces[i, 1] -= wall_repulsion * (y - (1 - r))

            # Gravity towards center
            forces[i] -= gravity * (centers[i] - 0.5)

        # Update positions using velocity Verlet integration
        velocity = dampening * (velocity + forces * dt)
        centers += velocity * dt

        # Add random perturbation (simulated annealing)
        if iter % 20 == 0:
            centers += np.random.normal(0, 0.004 * temperature, centers.shape) # Reduced magnitude

        # Keep circles within bounds
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])

    return centers, radii


def optimize_stage2(centers, radii, iterations):
    """
    Second optimization stage: Grow radii while maintaining valid packing
    """
    n = len(centers)
    growth_rate = 0.0008 # Reduced growth rate

    for _ in range(iterations):
        # Calculate overlaps
        overlaps = np.zeros(n)

        # Circle-circle overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    overlap = min_dist - dist
                    overlaps[i] += overlap
                    overlaps[j] += overlap

        # Wall overlaps
        for i in range(n):
            x, y = centers[i]
            r = radii[i]

            if x < r:
                overlaps[i] += r - x
            if x > 1 - r:
                overlaps[i] += x - (1 - r)
            if y < r:
                overlaps[i] += r - y
            if y > 1 - r:
                overlaps[i] += y - (1 - r)

        # Grow circles without overlaps
        for i in range(n):
            if overlaps[i] <= 1e-6:
                # Calculate available space
                min_dist_to_others = float('inf')
                for j in range(n):
                    if i != j:
                        dist = np.linalg.norm(centers[i] - centers[j]) - radii[j]
                        min_dist_to_others = min(min_dist_to_others, dist)

                space_to_walls = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])
                available_space = min(min_dist_to_others, space_to_walls) - radii[i]

                # Grow radius based on available space
                radii[i] += min(growth_rate, available_space * 0.6)
            else:
                # Shrink overlapping circles
                radii[i] = max(0.01, radii[i] - overlaps[i] * 0.1)

    return centers, radii


def optimize_stage3(centers, radii, iterations):
    """
    Third optimization stage: Fine-tune positions and sizes
    """
    n = len(centers)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)

    temperature = 0.01  # Initial temperature for simulated annealing
    cooling_rate = 0.995  # Cooling rate
    position_perturbation = 0.002
    radius_perturbation = 0.0008

    for iter in range(iterations):
        # Make a copy of current state
        new_centers = centers.copy()
        new_radii = radii.copy()

        # Randomly perturb positions and radius
        idx = np.random.randint(0, n)
        new_centers[idx] += np.random.normal(0, position_perturbation, 2)  # Reduced position perturbation
        radius_change = np.random.normal(0, radius_perturbation) # Reduced radius perturbation
        new_radii[idx] = max(0.01, new_radii[idx] + radius_change)

        # Keep circle within bounds
        new_centers[idx, 0] = np.clip(new_centers[idx, 0], new_radii[idx], 1 - new_radii[idx])
        new_centers[idx, 1] = np.clip(new_centers[idx, 1], new_radii[idx], 1 - new_radii[idx])

        # Check for overlaps - more efficient overlap check
        valid = True
        for i in range(n):
            if i == idx:
                continue
            dist_vec = new_centers[idx] - new_centers[i]
            dist_sq = np.sum(dist_vec * dist_vec)
            min_dist = new_radii[idx] + new_radii[i]
            if dist_sq < min_dist * min_dist: # Avoid sqrt
                valid = False
                break

        # Accept or reject based on simulated annealing
        if valid:
            new_sum = np.sum(new_radii)
            delta_e = new_sum - np.sum(radii)
            if delta_e > 0 or np.random.random() < np.exp(delta_e / temperature):
                centers = new_centers
                radii = new_radii

                if new_sum > best_sum:
                    best_sum = new_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()

        # Cool temperature
        temperature *= cooling_rate

        # Periodically optimize all radii
        if iter % 50 == 0:
            centers, radii = optimize_all_radii(centers, radii)
            sum_radii = np.sum(radii)
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_centers = centers.copy()
                best_radii = radii.copy()

    return best_centers, best_radii


def optimize_all_radii(centers, radii):
    """Optimize all radii to fill available space"""
    n = len(centers)

    # Calculate available space for each circle
    for _ in range(5):  # Reduced iterations for speed
        for i in range(n):
            # Calculate minimum distance to other circles
            min_dist_to_others = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j]) - radii[j]
                    min_dist_to_others = min(min_dist_to_others, dist)

            # Calculate minimum distance to walls
            space_to_walls = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])

            # Set radius to fill available space - softened update
            available_space = min(min_dist_to_others, space_to_walls)
            radii[i] = 0.8 * radii[i] + 0.2 * available_space # Dampened update

    return centers, radii


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