import numpy as np
import matplotlib.pyplot as plt

def parse_grid(file_path):
    """
    Parses a grid from a file.

    Args:
        file_path (str): The path to the file containing the grid.

    Returns:
        list of list of int: The parsed grid.
    """
    with open(file_path, 'r') as file:
        grid = []
        for line in file:
            grid.append([int(value) for value in line.split()])
    return grid

def visualize_path(grid, path):
    """
    Visualizes the given path on the grid using matplotlib.

    Args:
        grid (list of list of int): The grid representing the values.
        path (list of tuple of int): The path to be visualized.
    """
    rows, cols = len(grid), len(grid[0])
    path_set = set(path)  # For quick lookup of path cells

    grid_array = np.array(grid)
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(grid_array, cmap='Blues')

    # Overlay the path values
    for (i, j) in path:
        ax.text(j, i, f'{grid[i][j]}', va='center', ha='center', color='red', fontsize=12, weight='bold')
    
    # Add grid values for non-path cells
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in path_set:
                ax.text(j, i, f'{grid[i][j]}', va='center', ha='center', color='black')

    # Plot arrows to show the path direction
    for k in range(len(path) - 1):
        (x1, y1), (x2, y2) = path[k], path[k + 1]
        ax.annotate('', xy=(y2, x2), xytext=(y1, x1),
                    arrowprops=dict(facecolor='red', shrink=0.05, headwidth=8, width=2))

    plt.show()
