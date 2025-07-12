# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles in a unit square"""
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
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Initial guess for radii and centers
    radii[:] = 0.07

    # Optimized placement based on hexagonal and edge packing principles

    # Central hexagonal core (1 + 6 = 7 circles)
    centers[0] = [0.5, 0.5]
    centers[1] = [0.5 + 2 * radii[1], 0.5]
    centers[2] = [0.5 - 2 * radii[2], 0.5]
    centers[3] = [0.5 + radii[3], 0.5 + np.sqrt(3) * radii[3]]
    centers[4] = [0.5 + radii[4], 0.5 - np.sqrt(3) * radii[4]]
    centers[5] = [0.5 - radii[5], 0.5 + np.sqrt(3) * radii[5]]
    centers[6] = [0.5 - radii[6], 0.5 - np.sqrt(3) * radii[6]]

    # Second layer (6 circles, adjusted positions for better packing)
    layer2_radius = 0.22  # Adjusted distance from center
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6  # Offset for better staggering
        centers[7 + i] = [0.5 + layer2_radius * np.cos(angle), 0.5 + layer2_radius * np.sin(angle)]
        radii[7 + i] = 0.065 #slightly smaller for tighter pack

    # Third layer (6 circles, placed strategically near corners)
    centers[13] = [0.1, 0.1]
    centers[14] = [0.9, 0.1]
    centers[15] = [0.1, 0.9]
    centers[16] = [0.9, 0.9]
    centers[17] = [0.3, 0.1]
    centers[18] = [0.1, 0.3]

    # Fourth layer (7 circles, placed strategically near edges)
    centers[19] = [0.7, 0.1]
    centers[20] = [0.9, 0.3]
    centers[21] = [0.3, 0.9]
    centers[22] = [0.1, 0.7]
    centers[23] = [0.9, 0.7]
    centers[24] = [0.7, 0.9]
    centers[25] = [0.5, 0.9]


    radii[13:] = 0.055 # smaller radius for external circles

    # Ensure all circles are inside the unit square
    centers = np.clip(centers, 0.001, 0.999)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
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

    # First, initialize radii based on distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively adjust radii to prevent overlaps
    converged = False
    while not converged:
        converged = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                max_sum = dist

                # If current radii would cause overlap
                if radii[i] + radii[j] > max_sum:
                    # Reduce the larger radius to prevent overlap
                    reduction = (radii[i] + radii[j] - max_sum) / 2.0
                    if radii[i] > radii[j]:
                        old_radius = radii[i]
                        radii[i] -= reduction
                        if abs(old_radius - radii[i]) > 1e-8:
                            converged = False
                    else:
                        old_radius = radii[j]
                        radii[j] -= reduction
                        if abs(old_radius - radii[j]) > 1e-8:
                            converged = False

                    radii[i] = max(radii[i], 0.0001)
                    radii[j] = max(radii[j], 0.0001)

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
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)