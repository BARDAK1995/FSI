# Define the function to find indices of a point in the domain with updated domain boundaries
# Correcting the function to properly calculate the number of cells in the x-direction based on the x-axis length
from typing import List, Tuple
import matplotlib.pyplot as plt

import numpy as np

def find_indices_corrected_v2(domain_x_min: float, domain_x_max: float, domain_y_max: float, num_cells_y: int, point_x: float, point_y: float) -> Tuple[int, int]:
    # Calculate the domain lengths
    domain_x = domain_x_max - domain_x_min
    domain_y = domain_y_max  # y starts from 0
    
    # Calculate the cell size in each dimension
    cell_size_y = domain_y / num_cells_y
    cell_size_x = cell_size_y  # Assuming square cells, so using num_cells_y for x-axis as well

    # Calculate the minimum coordinate for x and y
    min_x = domain_x_min
    min_y = 0  # y starts from 0
    
    # Calculate the indices
    index_x = int((point_x - min_x) // cell_size_x)  # Correcting the index by multiplying by 2
    index_y = int((point_y - min_y) // cell_size_y)
    
    return index_x, index_y

# Corrected wrapper function to handle an array of points and also plot them
def find_indices_for_points_and_plot_corrected_v2(domain_x_min: float, domain_x_max: float, domain_y_max: float, num_cells_y: int, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    indices = []
    for i, point in enumerate(points):
        point_x, point_y = point
        index_x, index_y = find_indices_corrected_v2(domain_x_min, domain_x_max, domain_y_max, num_cells_y, point_x, point_y)
        indices.append((index_x, index_y))
    
    # Plotting setup is the same as before
    plt.figure(figsize=(10, 5))
    plt.xlim(domain_x_min, domain_x_max)
    plt.ylim(0, domain_y_max)
    
    domain_x = domain_x_max - domain_x_min
    domain_y = domain_y_max  # y starts from 0
    
    cell_size_x = domain_x / num_cells_y
    cell_size_y = domain_y / num_cells_y
    
    for i in np.arange(domain_x_min, domain_x_max + cell_size_x, cell_size_x):
        plt.axvline(i, color='gray', linewidth=0.5)
    for i in np.arange(0, domain_y_max + cell_size_y, cell_size_y):
        plt.axhline(i, color='gray', linewidth=0.5)
    
    x_coords, y_coords = zip(*points)
    plt.scatter(x_coords, y_coords, color='red', zorder=5)
    
    for i, point in enumerate(points):
        # plt.annotate(f"{indices[i]}", (point[0], point[1]), textcoords="offset points", xytext=(0, -10), ha='center')
        plt.annotate(f"{i+1}", (point[0], point[1]), textcoords="offset points", xytext=(0, 10), ha='center',fontsize=13)

    plt.xlabel('X Coordinate (mm)')
    plt.ylabel('Y Coordinate (mm)')
    plt.title('Probe locations',fontsize=20)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()
    
    return indices

# Define the updated domain dimensions and the number of cells in the y-direction

domain_x_min = -4  # mm

domain_x_max = 16   # mm

domain_y_max = 5   # mm
num_cells_y = 100

# setup1
# points = [(-3, 2.5), (-1, 2.5), (1, 2.5), (3, 2.5), (5, 2.5), (7, 2.5), (9, 2.5), (11, 2.5), (13, 2.5), (15, 2.5)]
# setup2
points = [(1, 0.07), (2, 0.07), (3, 0.07), (4, 0.07), (5, 0.07), (6, 0.07), (7, 0.07), (8, 0.07), (9, 0.07), (10, 0.07)]
# setup3
# points = [(5, 0.57), (5.5, 0.57), (6, 0.57), (6.5, 0.57), (7, 0.57), (7.5, 0.57), (8, 0.57), (8.5, 0.57), (9, 0.57), (9.5, 0.57)]


# Run the corrected function
indices_with_plot_corrected_v2 = find_indices_for_points_and_plot_corrected_v2(domain_x_min, domain_x_max, domain_y_max, num_cells_y, points)
indices_with_plot_corrected_v2


