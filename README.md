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

Conceptually, this orienteering problem can be solved using four different approaches: with (1) brute-force, a greedy algorithm, dynamic programming, or a (meta)Heuristic algorithm (e.g. genetic algorithms). Each method has its benefits and downsides, especially when considering large grids and large T.

## 1. Brute Force (Depth-First Search or Breadth-First Search)
Depth-First Search (DFS): Explores each path to its fullest depth before backtracking. Breadth-First Search (BFS): Explores all paths level by level.
Benefits:
- Guarantees finding the optimal solution.
- Simple to implement and understand.
Downsides:
- Exponential time complexity: Very slow for large grids (N x N) and large T.
- DFS may use less memory but can go very deep, while BFS may use a lot of memory due to storing all nodes at each level.
## 2. Greedy Algorithm
The simplest implementation is that, at each step, the drone moves to the adjacent cell with the highest current value.
Benefits:
- Very fast and requires minimal computation.
- Simple to implement and can quickly provide a solution.
Downsides:
- Does not guarantee the optimal solution.
- Can get stuck in local optima and miss higher-value paths available through more complex routes.
## 3. Dynamic Programming
Description:
- Breaks the problem into subproblems and solves each subproblem only once, storing the results for future use.
- Can be implemented using a table to store the maximum value collected up to each cell for a given time step.
Benefits:
- More efficient than brute force for larger grids as it avoids redundant calculations.
- Provides an optimal solution by considering all possible paths.
Downsides:
- Still has significant computational complexity, though it is more feasible than brute force for larger grids.
- Can consume a lot of memory, especially for large N and T.
## 4. (Meta)Heuristic Algorithm (Genetic Algorithm)
Uses a population of solutions (paths) and evolves them over generations using crossover, mutation, and selection to find high-quality solutions.
Benefits:
- Can handle very large search spaces more efficiently than brute force.
- Does not require an exhaustive search and can provide good solutions within the time limit T.
- Can avoid local optima better than greedy algorithms.
Downsides:
- Does not guarantee finding the optimal solution due to stochastic nature.
- Requires careful tuning of parameters (population size, mutation rate, etc.).
- Computationally expensive compared to greedy algorithms but usually more efficient than brute force and dynamic programming for very large problems.

## Comparison for Large Grids and Large T:
Brute Force: Impractical due to exponential growth in computation time with grid size and time steps.
Greedy Algorithm: Fast but potentially far from optimal, especially in large grids.
Dynamic Programming: Provides optimal solutions but may become impractical due to memory and computation time constraints in very large grids and long time horizons.
Genetic Algorithm: Balances between solution quality and computational efficiency. More suitable for large grids and long time horizons where exact methods become infeasible.

## Conclusion:
For very large grids and significant time steps (large N and T), metaheuristic approaches like Genetic Algorithms are often the most practical choice due to their ability to provide good solutions within a reasonable time frame. Greedy algorithms can be used for quick, approximate solutions, while dynamic programming can be leveraged for medium-sized problems where exact solutions are still feasible. Brute force methods are typically only useful for small instances or as a benchmark for solution quality.

## Implementation steps

## Testing and Validation
- Test Cases: developing test cases using provided example levels of different sizes to validate algorithm.
- Edge Cases: considering edge cases such as minimal grid size, maximal time steps, and extreme starting positions.
- Performance Evaluation: assessing the algorithm's performance with larger grid sizes and time steps to ensure scalability.

# Code implementation



# Results and evaluation


# Suggestions for further research
