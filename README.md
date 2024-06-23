# drone_pathfinder
Path-planning algorithm implementation that maximizes the value collected by a single drone navigating a dynamic grid within a time limit

# Content


# Summary


# Problem decsription and requirements

In this project, we are working on a simulation in which a drone is capable of independently monitoring an area. This project focuses on a single drone navigating a (N x N) grid where each cell has a numerical value. The drone must find the most optimal path to fly over, i.e. a path that maximizes the total sum of all collected values, while at the same time taking into consideration the dynamic nature of the cell values.

## Requirements

The main goal is to develop an algorithm that returns the most valuable path and total score within the constraints, and should be efficient for larger values of N and t.

## Deliverables
- Source code of the path-planning algorithm.
- Documentation explaining the methodology, choices made, and any assumptions.
- A demonstration of the algorithm using the provided example levels of different sizes.

## Input parameters

- N: Size of the grid (N x N)
- t: Total number of discrete time steps
- T: Maximum time duration for the algorithm in milliseconds
- (x, y): Starting position of the drone in the grid
- r: the speed at which cell values "regenerate"

## Constraints

- The drone moves to adjacent cells in one time step (horizontal, vertical, or diagonal).
- The entire grid is visible to the drone.
- The presence of the drone in a cell collects the value of that cell.
- Values of cells increase over time after being visited (with regeneration speed r defined by the user).
- The algorithm must complete within the given time limit T.

## Evaluation Criteria

- Code quality and efficiency.
- Explanation and justification of the chosen methodology.
- Ability to handle larger grid sizes and time steps within the time limit.
- Clarity and thoroughness of the documentation.


# Problem analysis

# Algorithm design

## Path-planning strategy

## Implementation steps

## Testing and Validation
- Test Cases: developing test cases using provided example levels of different sizes to validate algorithm.
- Edge Cases: considering edge cases such as minimal grid size, maximal time steps, and extreme starting positions.
- Performance Evaluation: assessing the algorithm's performance with larger grid sizes and time steps to ensure scalability.

# Code implementation



# Results and evaluation


# Suggestions for further research
