# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Central circle
    centers[0] = [0.5, 0.5]
    radii[0] = 0.21  # Start with a slightly larger radius

    # Inner hexagon (6 circles)
    r1 = 0.23
    for i in range(6):
        angle = 2 * np.pi * i / 6
        centers[i + 1] = [0.5 + r1 * np.cos(angle), 0.5 + r1 * np.sin(angle)]
        radii[i + 1] = 0.125 # slight increase in radius

    # Middle ring (12 circles) - slightly distorted hexagon
    r2 = 0.42
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        centers[i + 7] = [0.5 + r2 * np.cos(angle), 0.5 + r2 * np.sin(angle)]
        radii[i + 7] = 0.09 # Adjust radius, slightly larger

    # Corners (4 circles)
    corner_dist = 0.09
    centers[19] = [corner_dist, corner_dist]
    centers[20] = [1 - corner_dist, corner_dist]
    centers[21] = [corner_dist, 1 - corner_dist]
    centers[22] = [1 - corner_dist, 1 - corner_dist]
    radii[19:23] = 0.085 #radius for corners, slightly larger

    # Edges (4 circles)
    edge_offset = 0.06
    centers[23] = [0.5, edge_offset]
    centers[24] = [0.5, 1 - edge_offset]
    centers[25] = [edge_offset, 0.5]
    centers[26 - 1] = [1 - edge_offset, 0.5]
    radii[23:] = 0.065 # radius for edges, slightly larger


    # Iterative refinement of radii and positions
    for _ in range(75): # increased iterations for better convergence
        for i in range(n):
            # Calculate distance to walls
            dist_to_walls = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])

            # Calculate distance to other circles
            dist_to_circles = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    dist_to_circles = min(dist_to_circles, dist - radii[j])

            # Update radius - limit increase per iteration, more aggressively
            radii[i] = min(dist_to_walls, dist_to_circles, radii[i] * 1.07) # increased growth factor

        # Adjust positions to reduce overlaps.  More subtle adjustments
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    overlap = radii[i] + radii[j] - dist
                    if overlap > 0:
                        # Move circles apart proportionally to their radii, with a smaller factor
                        move_x = (centers[i][0] - centers[j][0]) * overlap * 0.007 # increased movement factor
                        move_y = (centers[i][1] - centers[j][1]) * overlap * 0.007 # increased movement factor
                        centers[i][0] += move_x
                        centers[i][1] += move_y
                        centers[j][0] -= move_x
                        centers[j][1] -= move_y

                        # Keep circles within the unit square
                        centers[i][0] = np.clip(centers[i][0], radii[i], 1 - radii[i])
                        centers[i][1] = np.clip(centers[i][1], radii[i], 1 - radii[i])
                        centers[j][0] = np.clip(centers[j][0], radii[j], 1 - radii[j])
                        centers[j][1] = np.clip(centers[j][1], radii[j], 1 - radii[j])


    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, iteratively adjust radii to avoid overlaps
    # This approach is more accurate than the proportional scaling
    # We use multiple iterations to converge to a valid solution
    for _ in range(15):  # Multiple iterations for better convergence
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    max_allowed = max(0.0, dist - radii[j])
                    radii[i] = min(radii[i], max_allowed)

    return radii

# EVOLVE-BLOCK-END


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