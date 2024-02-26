import numpy as np

def generate_points(start, end, n):
    """
    Generates n linearly spaced points between two end locations with x and y coordinates.
    
    Args:
    - start: A tuple representing the starting coordinates (x, y).
    - end: A tuple representing the ending coordinates (x, y).
    - n: The number of points to generate.
    
    Returns:
    - A list of n linearly spaced points between start and end.
    """
    x_values = np.linspace(start[0], end[0], n)
    y_values = np.linspace(start[1], end[1], n)
    return list(zip(x_values, y_values))

def write_points_to_file(filename, points):
    """
    Writes the generated points to a file in the specified format.
    
    Args:
    - filename: Name of the file to write the points to.
    - points: A list of (x, y) coordinates to write to the file.
    """
    with open(filename, 'w') as file:
        # Write the number of points at the top of the file
        file.write(f"{len(points)}\n")
        for x, y in points:
            # Write each point in the format: x y
            file.write(f"{x:.4f} {y:.4f}\n")

# Parameters
n_points = 90
start = (-0.003, 0.002)
end = (0.015, 0.002)
# Generate the points
points = generate_points(start, end, n_points)

# Write the points to a file
filename = "probes.data"
write_points_to_file(filename, points)

print(f"{n_points} points have been generated and written to {filename}.")
