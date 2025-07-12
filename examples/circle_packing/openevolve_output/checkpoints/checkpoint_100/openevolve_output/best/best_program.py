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
    
    # Initialize with a carefully designed pattern
    # This pattern is based on research into optimal circle packings
    centers, radii = initialize_specialized_pattern(n)
    
    # First optimization phase: local adjustments with fixed radii
    centers = optimize_positions(centers, radii, iterations=150)
    
    # Second optimization phase: grow radii while maintaining validity
    centers, radii = optimize_radii(centers, radii, iterations=200)
    
    # Third optimization phase: fine-tuning with variable radii
    centers, radii = fine_tune(centers, radii, iterations=150)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def initialize_specialized_pattern(n):
    """
    Initialize with a specialized pattern designed for n=26 circles.
    The pattern uses a combination of:
    - A central larger circle
    - Concentric rings with different-sized circles
    - Strategic placement at corners and edges
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Central circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.16
    
    # First ring (6 circles)
    r1 = 0.28
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers[i + 1] = [0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)]
        radii[i + 1] = 0.11
    
    # Second ring (12 circles)
    r2 = 0.46
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12  # Offset to stagger
        centers[i + 7] = [0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)]
        radii[i + 7] = 0.09
    
    # Corner circles (4)
    corner_offset = 0.11
    centers[19] = [corner_offset, corner_offset]
    centers[20] = [1 - corner_offset, corner_offset]
    centers[21] = [corner_offset, 1 - corner_offset]
    centers[22] = [1 - corner_offset, 1 - corner_offset]
    radii[19:23] = 0.09
    
    # Edge circles (4)
    edge_offset = 0.09
    centers[23] = [0.5, edge_offset]
    centers[24] = [0.5, 1 - edge_offset]
    centers[25] = [edge_offset, 0.5]
    centers[26 - 1] = [1 - edge_offset, 0.5]
    radii[23:] = 0.08
    
    return centers, radii


def optimize_positions(centers, radii, iterations=150):
    """
    Optimize circle positions while keeping radii fixed.
    Uses a physics-based approach with repulsive forces.
    """
    n = len(centers)
    learning_rate = 0.005
    
    for _ in range(iterations):
        # Compute repulsive forces between circles
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = centers[i] - centers[j]
                    dist = np.sqrt(np.sum(diff**2))
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        # Normalize direction and scale by overlap
                        direction = diff / (dist + 1e-10)
                        overlap = min_dist - dist
                        forces[i] += direction * overlap * 10.0
            
            # Add wall repulsion forces
            x, y = centers[i]
            r = radii[i]
            
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 10.0
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 10.0
        
        # Update positions
        centers += forces * learning_rate
        
        # Ensure circles stay within bounds
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])
    
    return centers


def optimize_radii(centers, radii, iterations=200):
    """
    Optimize circle radii while adjusting positions to maintain validity.
    Uses a growth-based approach with collision resolution.
    """
    n = len(centers)
    growth_rate = 0.001
    position_adjustment_rate = 0.005
    
    for _ in range(iterations):
        # Grow all radii by a small amount
        radii += growth_rate
        
        # Check for collisions and resolve them
        for _ in range(5):  # Multiple resolution passes per growth step
            # Compute collisions
            collisions = False
            forces = np.zeros((n, 2))
            
            # Circle-circle collisions
            for i in range(n):
                for j in range(i+1, n):
                    diff = centers[i] - centers[j]
                    dist = np.sqrt(np.sum(diff**2))
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        collisions = True
                        # Calculate repulsive force
                        direction = diff / (dist + 1e-10)
                        overlap = min_dist - dist
                        force = direction * overlap * 0.5
                        forces[i] += force
                        forces[j] -= force
            
            # Wall collisions
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                
                if x < r:
                    collisions = True
                    forces[i, 0] += (r - x)
                if x > 1 - r:
                    collisions = True
                    forces[i, 0] -= (x - (1 - r))
                if y < r:
                    collisions = True
                    forces[i, 1] += (r - y)
                if y > 1 - r:
                    collisions = True
                    forces[i, 1] -= (y - (1 - r))
            
            # If no collisions, we're done with this resolution pass
            if not collisions:
                break
                
            # Apply forces to resolve collisions
            centers += forces * position_adjustment_rate
            
            # If we still have collisions after all passes, shrink the radii slightly
            if collisions and _ == 4:
                radii *= 0.995
    
    # Final adjustment to ensure no overlaps
    centers, radii = ensure_no_overlaps(centers, radii)
    
    return centers, radii


def fine_tune(centers, radii, iterations=150):
    """
    Fine-tune the packing by allowing individual radii to grow
    based on available space, focusing on maximizing total sum.
    """
    n = len(centers)
    
    for _ in range(iterations):
        # For each circle, try to grow it based on available space
        for i in range(n):
            # Calculate minimum distance to other circles
            min_dist_to_circle = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    gap = dist - (radii[i] + radii[j])
                    min_dist_to_circle = min(min_dist_to_circle, gap)
            
            # Calculate distance to walls
            x, y = centers[i]
            dist_to_walls = min(x - radii[i], y - radii[i], 
                               1 - x - radii[i], 1 - y - radii[i])
            
            # Grow radius by a fraction of available space
            available_space = min(min_dist_to_circle, dist_to_walls)
            if available_space > 0:
                radii[i] += available_space * 0.3
        
        # Ensure no overlaps after growth
        centers, radii = ensure_no_overlaps(centers, radii)
        
        # Small position adjustments to maximize space utilization
        centers = optimize_positions(centers, radii, iterations=5)
    
    return centers, radii


def ensure_no_overlaps(centers, radii):
    """
    Ensure there are no overlaps between circles or with walls.
    If overlaps exist, shrink radii slightly until resolved.
    """
    n = len(centers)
    max_iterations = 20
    shrink_factor = 0.99
    
    for _ in range(max_iterations):
        overlaps = False
        
        # Check circle-circle overlaps
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j]:
                    overlaps = True
                    # Shrink both circles proportionally
                    ratio = dist / (radii[i] + radii[j] + 1e-10)
                    radii[i] *= ratio * 0.99
                    radii[j] *= ratio * 0.99
        
        # Check wall overlaps
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            if x < r or x > 1 - r or y < r or y > 1 - r:
                overlaps = True
                # Move circle inside and shrink if necessary
                centers[i, 0] = np.clip(x, r, 1 - r)
                centers[i, 1] = np.clip(y, r, 1 - r)
                radii[i] *= shrink_factor
        
        if not overlaps:
            break
    
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