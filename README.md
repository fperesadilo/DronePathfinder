![alt text](https://github.com/fperesadilo/drone_pathfinder/blob/main/img/img.png)



# Summary
In this project, I developed a simulation in which a drone independently monitors an area. The drone navigates a grid to find the most optimal path that maximizes the total sum of collected values, considering a dynamic nature of cell values (regeneration of value after node visit). This repo presents two algorithms of solving this orienteering problem: an approach using (1) dynamic programming, and (2) a genetic algorithm. Dynamic programming was selected for its efficiency in finding optimal solutions in smaller to medium-sized grids within feasible time limits. It systematically breaks down the problem into subproblems and stores intermediate results. A genetic algorithm was chosen as a metaheuristic approach capable of handling larger search spaces and longer time horizons where exact methods like DP might become impractical due to computational and memory constraints.

# Content
- Problem description and requirements
- Problem analysis
- Algorithm design
- Implementation steps
- Testing and validation
- Code usage and examples (incl. visualisation)
- Results and evaluation

# Problem description and requirements

In this project, I have been working on a simulation in which a drone is capable of independently monitoring an area. This project focuses on a single drone navigating a (N x N) grid where each cell has a numerical value. The drone must find the most optimal path to fly over, i.e. a path that maximizes the total sum of all collected values, while at the same time taking into consideration the dynamic nature of the cell values.

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

Before I start conceptualizing solutions for the problem, I tend to look how smarter people than me look at the problem or how they define it, or whether I can reduce the problem to an existing problem and model it from there. In the literature, this problem is called the Orienteering Problem (OP), also called the Selective Travelling Salesperson Problem or the Maximum Collection Problem, and is a well known NP-hard combinatorial optimisation problem ([source](https://bird.bcamath.org/bitstream/handle/20.500.11824/730/ea4op.pdf)). The origin of the problem is a sports game, where the participants are given a topographical map with detailed control points with rewards. The participants try to visit the points in order to maximise the total prize obtained; however, there is a time limitation. Therefore, the OP is basically a routing problem with profits and it can be seen as a combination between the Knapsack Problem and the Travelling Salesperson Problem (TSP) 

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

- Brute Force: Impractical due to exponential growth in computation time with grid size and time steps.
- Greedy Algorithm: Fast but potentially far from optimal, especially in large grids.
- Dynamic Programming: Provides optimal solutions but may become impractical due to memory and computation time constraints in very large grids and long time horizons.
- Genetic Algorithm: Balances between solution quality and computational efficiency. More suitable for large grids and long time horizons where exact methods become infeasible.

## Conclusion:
For very large grids and significant time steps (large N and T), metaheuristic approaches like Genetic Algorithms are often the most practical choice due to their ability to provide good solutions within a reasonable time frame. Greedy algorithms can be used for quick, approximate solutions, while dynamic programming can be leveraged for medium-sized problems where exact solutions are still feasible. Brute force methods are typically only useful for small instances or as a benchmark for solution quality.

## Implementation steps

In this case, I have decided only to implement two algorithms using the dynamic programming (DP) approach and a genetic algorithm (GA).

The **dynamic programming algorithm (DP)** for path planning in a grid with changing cell values works like a methodical treasure hunt. Imagine starting at a point on a map where each spot has a different value. The goal is simple: find a path that, over a set number of moves, collects the most value possible.

To achieve this, the algorithm uses a table that keeps track of the best possible value for reaching each spot on the map at each step of the journey. As it progresses through each step, it updates this table to ensure it always knows the best way to maximize the total value collected up to that point.

The algorithm also considers the cost of revisiting spots where values might change over time. It carefully evaluates each move—whether to revisit a known spot or explore new ones—ensuring it makes the best decisions based on the potential rewards at each step.

Overall, the algorithm combines systematic exploration with smart decision-making to find a path that not only respects time limits but also gathers the highest possible total value. It's like navigating a map to gather as much treasure as possible, using a step-by-step approach to ensure every move counts towards maximizing the loot.

Pseudocode for the dynammic programming approach:
```
Class DynamicProgramming:
    Initialize with grid, start, T, regen_value
    Initialize dp table to store maximum values
    Initialize parent table to store paths
    Define possible move directions

    Function reconstruct_path(x, y, T):
        Initialize empty path
        While T > 0:
            Add (x, y) to path
            Update (x, y) to parent cell
            Decrement T
        Add start position to path
        Reverse path
        Return path

    Function max_path_value_and_path():
        Set initial value at start position in dp table
        For each time step t from 1 to T:
            For each cell (x, y) in the grid:
                Get current value from dp table
                For each move direction (dx, dy):
                    Calculate new position (nx, ny)
                    If (nx, ny) is within grid:
                        If (nx, ny) was visited:
                            Apply regeneration cost
                        Else:
                            Calculate new value
                        If new value is better, update dp and parent tables
        Find maximum value and reconstruct path from start position
        Return maximum value, path, dp table, parent table, and elapsed time

```

The **genetic algorithm (GA)** for path planning in a grid with changing cell values operates very much like evolutionary principles in nature. In this algorithm, multiple potential paths (represented as chromosomes) are generated randomly. Each path's fitness, or how well it collects values, is evaluated. Like natural selection, paths with higher fitness are more likely to "survive" and influence future generations of paths.

Through a process resembling genetic crossover and mutation, better-performing paths are combined and altered over successive generations to explore new potential paths. This mimics how genetic traits combine and mutate in living organisms, fostering diversity and potentially uncovering better solutions.

Crossover involves combining two parent paths to create offspring paths. In this algorithm, crossover determines how parts of two parent paths are exchanged to create new paths. For example, segments of one parent's path may replace corresponding segments of another parent's path. This process aims to retain and combine successful traits from both parents, potentially producing offspring paths that inherit beneficial characteristics.

Mutation introduces random changes in individual paths. In orienteering, this means altering certain moves within a path randomly. Mutation helps explore new paths that may not be present in the current population, thus promoting diversity and preventing the algorithm from getting stuck in local optima. It allows for occasional exploration of paths with unexpected but potentially higher values, contributing to the algorithm's ability to find better solutions over time.

The algorithm iteratively refines these paths, balancing exploration (searching for new, potentially better paths) and exploitation (improving known paths). By continually evolving and selecting the most promising solutions, it seeks to converge on a path that maximizes value collection within the defined constraints.

Ultimately, the genetic algorithm navigates the trade-offs between exploration and exploitation to find a high-value path efficiently. It harnesses the power of evolutionary principles to adapt and improve paths over time, aiming for optimal performance in path planning scenarios. The algorithm is inspired by the implementation by [mc-ride](https://github.com/mc-ride/orienteering).

Pseudocode for the genetic algorithm:

```
Class GeneticAlgorithm:
    Initialize with grid, start_point, end_point, tmax, regen_value, popsize, genlimit, kt, isigma, msigma, mchance, elitismn

    Function fitness(chrom):
        Initialize path, collected_values, current_pos, total_value
        For each move in chrom:
            Calculate next_pos
            If next_pos is within grid:
                Update current_pos and path
                If current_pos not visited:
                    Update total_value and collected_values
        If current_pos is not end_point:
            Set total_value to 0
        Return total_value and path

    Function crossover(c1, c2):
        Choose a crossover point
        Combine parts of c1 and c2 to create offspring
        Return offspring

    Function mutate(chrom):
        For each move in chrom:
            Possibly change move based on mutation chance
        Return mutated chromosome

    Function run_algorithm():
        Initialize population with random chromosomes and their fitness
        For each generation up to genlimit:
            Create next generation
            For each individual in population:
                Select parents
                Create offspring with crossover and mutation
                Evaluate fitness of offspring
                Maintain elitism in population
            Update population with next generation
        Return the best path and its fitness
```

Here, **elitism** refers to a strategy where the best-performing individuals (paths) from one generation are directly carried over to the next generation without undergoing crossover or mutation. These elite individuals represent the highest fitness solutions found so far and are preserved to ensure that the best solutions discovered are not lost in subsequent generations.

Elitism helps maintain the overall quality of solutions across generations by preventing the algorithm from losing the best-performing paths due to crossover or mutation operations that might inadvertently reduce their fitness.

## Testing and Validation
In this project, I implemented a testing and validation framework to ensure the correctness and robustness of my algorithms. Here is a brief summary:

Dynamic Programming Class Tests
- Path Length Matches T: Verified that the algorithm returns a path of length equal to T by initializing a grid and checking the path length.
- Positive Collected Value: Ensured the maximum path value is positive by using a grid with negative values and checking if the collected value is greater than zero.
- Start Point Equals End Point: Confirmed the path starts and ends at the specified starting point by initializing a grid and verifying the first and last points of the path.
- Invalid Start Point: Tested handling of invalid starting points by providing an out-of-bounds start point and verifying that an IndexError is raised.

Genetic Algorithm Class Tests
- Path Length Matches tmax: Verified that the algorithm returns a path of length equal to tmax by initializing a grid and checking the path length.
- Positive Collected Value: Ensured the maximum fitness value is positive by using a grid with negative values and checking if the fitness value is greater than zero.
- Start Point Equals End Point: Confirmed the path starts and ends at the specified starting point by initializing a grid and verifying the first and last points of the path.
- Small Population and Generation Limit: Tested the algorithm with a small population size and limited generations by initializing a grid and checking if a valid path is returned.
- Mutation Probability: Verified mutation probability by running the algorithm multiple times with a high mutation chance and checking for different resulting paths.

# Code implementation and usage

The code is used as follows:

```python
# Imports
from orienteering import DynamicProgramming, GeneticAlgorithm
from utils import parse_grid, visualize_path

# Parse grids from provided txt files
grid_20 = parse_grid('grids/20.txt')

# Initialize input parameters
start = (10, 10)
T = 25
regen_value = 0

# Create an instance of DP implementation and calculate the maximum path value and path
path_finder = DynamicProgramming(grid_20, start, T, regen_value)
max_value, path, dp, parent, running_time_ms = path_finder.max_path_value_and_path()

# Print the results
print("Max Value:", max_value)
print("Path:", path)
visualize_path(grid_20, path)

# Initialize and run the genetic algorithm
ga_solver = GeneticAlgorithm(grid_20, start, end, T, regen_value)
best_path, best_value, running_time_ms = ga_solver.run_algorithm()

print("Best value:", best_value)
print("Elapsed time (in ms):", running_time_ms)
print("Best path:", best_path)

visualize_path(grid_20, best_path)
```

Visualisation result:

![alt text](https://github.com/fperesadilo/drone_pathfinder/blob/main/img/generated_path.png)

# Results and evaluation

In order to assess the efficiency of both algorithms, I have iterated over 50 predefined starting points on a predefined 1000x1000 grid, and computed the most valuable paths in 25 steps.

```python
# Initialize input parameters
start = (500, 500)
T = 25
regen_value = 1

# List to store the results
results_dp = []
results_ga = []

# Iterate over the starting points and collect running times
for start in starting_points_1000:
    # Performance DP
    dp_path_finder = DynamicProgramming(grid_1000, start, T, regen_value)
    max_value, path, dp, parent, elapsed_time = dp_path_finder.max_path_value_and_path()
    results_dp.append({'max_value': max_value, 'elapsed_time': elapsed_time})
    # Performance GA
    ga_path_finder = GeneticAlgorithm(grid_1000, start, start, T, regen_value)
    path, max_value, elapsed_time = ga_path_finder.run_algorithm()
    results_ga.append({'max_value': max_value, 'elapsed_time': elapsed_time})
```

The results:

![alt text](https://github.com/fperesadilo/drone_pathfinder/blob/main/img/efficiency_analysis.png)

Analysis:

- The dynamic programming algorithm is on average much faster than the GA
- The dynamic programming algorithm returns on average a better (i.e. higher value) path
- A possible explanation is that the parameters of the GA are not explored properly. Technically, the complexity for the GA increases linearly in T, while DP increases linearly in T^2 (depending on the grid size).

- **To-do:** add visualisation of path values 

# Suggestions for further research

- **To-do:** add conceptual solutions based on Ant Colony Optimization, and tebu search.
