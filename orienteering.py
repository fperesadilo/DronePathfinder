import time
import numpy as np
import random


class DynamicProgramming:
    """
    This class helps find the path with the maximum value within a given time limit (T) 
    in a grid where cells can be revisited but have a regeneration cost.

    ...

    Attributes:
        grid: A 2D list representing the grid, where each cell contains a value.
        start: A tuple representing the starting coordinates (x, y) in the grid.
        T: The maximum time allowed for the path.
        regen_value: The cost associated with revisiting a cell.
        rows: The number of rows in the grid.
        cols: The number of columns in the grid.
        dp: A 3D list used for dynamic programming to store the maximum value achievable 
             at each cell at a given time.
        parent: A 3D list used to store the parent cell for each cell in the optimal path 
             at a given time.
        directions: A list of tuples representing the possible moves in the grid 
             (-1, 1), (-1, 0), etc. represent up-right, up, etc.
    """
    def __init__(self, grid, start, T, regen_value):
        """
        Initialize the GridPathFinder with the given grid, start position, 
        maximum time T, and regeneration value.

        Args:
            grid (list of list of int): The grid representing the values.
            start (tuple of int): The starting position in the grid.
            T (int): The maximum time steps allowed.
            regen_value (int): The regeneration value when revisiting cells.
        """
        self.grid = grid
        self.start = start
        self.T = T
        self.regen_value = regen_value
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.dp = [[[-float('inf')] * (T + 1) for _ in range(self.cols)] for _ in range(self.rows)]
        self.parent = [[[(None, None)] * (T + 1) for _ in range(self.cols)] for _ in range(self.rows)]
        self.directions = [(-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)]

    def reconstruct_path(self, x, y, T):
        """
        Reconstruct the path from the given position (x, y) and time T 
        back to the start position.

        Args:
            x (int): The x-coordinate of the end position.
            y (int): The y-coordinate of the end position.
            T (int): The time step at the end position.

        Returns:
            list of tuple of int: The path from the start to the end position.
        """
        path = []
        current_x, current_y, current_t = x, y, T

        while current_t > 0:
            path.append((current_x, current_y))
            current_x, current_y = self.parent[current_x][current_y][current_t]
            current_t -= 1

        path.append((self.start[0], self.start[1]))
        path.reverse()

        return path

    def max_path_value_and_path(self):
        """
        Calculate the maximum path value and the path using dynamic programming.

        Returns:
            tuple: A tuple containing the maximum path value, the path,
                   the dp table, and the parent table.
        """
        # Initialize the starting position with the initial grid value
        self.dp[self.start[0]][self.start[1]][0] = self.grid[self.start[0]][self.start[1]]

        tic = time.time()
        
        # Iterate over each time step
        for t in range(1, self.T + 1):
            # Iterate over the subset of the grid within the T/2 distance from the start
            for x in range(max(0, self.start[0] - self.T//2), min(self.rows, self.start[0] + self.T//2 + 1)):
                for y in range(max(0, self.start[1] - self.T//2), min(self.cols, self.start[1] + self.T//2 + 1)):
                    current_value = self.dp[x][y][t-1]
                    # Consider all possible movements
                    for dx, dy in self.directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.rows and 0 <= ny < self.cols:
                            try:
                                # Reconstruct the path to check visited cells
                                visited_cells = self.reconstruct_path(x=x, y=y, T=t-1)
                                if (nx, ny) in visited_cells:
                                    # Apply regeneration if cell was already visited
                                    new_value = current_value + min(self.grid[nx][ny], visited_cells.index((nx, ny)) * self.regen_value)
                                else:
                                    new_value = current_value + self.grid[nx][ny]
                                # Update dp table and parent table if a better value is found
                                if new_value > self.dp[nx][ny][t]:
                                    self.dp[nx][ny][t] = new_value
                                    self.parent[nx][ny][t] = (x, y)
                            except Exception as e:
                                # Handle any exceptions silently
                                pass

        tac = time.time()
        elapsed_time_ms = (tac - tic) * 1000.0

        # Get the maximum value and the path from the start position
        max_value = self.dp[self.start[0]][self.start[1]][self.T]
        path = self.reconstruct_path(self.start[0], self.start[1], self.T)

        return max_value, path, self.dp, self.parent, elapsed_time_ms


class GeneticAlgorithm:
    """
    This class implements a genetic algorithm to find the path with the maximum value within a given time limit (T)
    in a grid environment, with moves represented as chromosomes.

    Attributes:
        grid (numpy.ndarray): A 2D array representing the grid, where each cell contains a value.
        start_point (tuple of int): The starting coordinates (x, y) in the grid.
        end_point (tuple of int): The ending coordinates (x, y) in the grid.
        tmax (int): The maximum time steps allowed for a path.
        popsize (int): The size of the population.
        genlimit (int): The maximum number of generations.
        kt (int): Number of tournaments to select parents.
        isigma (int): Sigma value for initial mutation.
        msigma (int): Sigma value for mutation.
        mchance (int): Chance of mutation.
        elitismn (int): Number of elitism.
    """
    def __init__(self, grid, start_point, end_point, tmax, regen_value, popsize=50, genlimit=1000, kt=5, isigma=2, msigma=1, mchance=2, elitismn=2):
        """
        Initialize the GeneticAlgorithm with parameters.

        Args:
            grid (numpy.ndarray): A 2D array representing the grid.
            start_point (tuple of int): Starting coordinates (x, y).
            end_point (tuple of int): Ending coordinates (x, y).
            tmax (int): Maximum time steps allowed for a path.
            popsize (int, optional): Population size for the genetic algorithm. Default is 50.
            genlimit (int, optional): Maximum number of generations. Default is 1000.
            kt (int, optional): Number of tournaments to select parents. Default is 5.
            isigma (int, optional): Sigma value for initial mutation. Default is 2.
            msigma (int, optional): Sigma value for mutation. Default is 1.
            mchance (int, optional): Chance of mutation. Default is 2.
            elitismn (int, optional): Number of elitism. Default is 2.
        """
        self.grid = np.array(grid)
        self.start_point = start_point
        self.end_point = end_point
        self.tmax = tmax
        self.regen_value = regen_value
        self.popsize = popsize
        self.genlimit = genlimit
        self.kt = kt
        self.isigma = isigma
        self.msigma = msigma
        self.mchance = mchance
        self.elitismn = elitismn
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def fitness(self, chrom):
        """
        Calculate the fitness of a chromosome (path) based on the grid.

        Args:
            chrom (list of tuple of int): Chromosome representing a path.

        Returns:
            tuple: A tuple containing the fitness value and the path.
        """
        path = [self.start_point]
        collected_values = set([self.start_point])
        current_pos = self.start_point
        total_value = self.grid[self.start_point]
        moves_left = self.tmax
    
        for move in chrom:
            if moves_left <= 0:
                break
            next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if 0 <= next_pos[0] < self.grid.shape[0] and 0 <= next_pos[1] < self.grid.shape[1]:
                current_pos = next_pos
                path.append(current_pos)
                if current_pos not in collected_values:
                    if current_pos in path[:-1]:  # Check if already visited
                        index = path.index(current_pos)
                        regen_cost = index * self.regen_value
                        total_value += min(self.grid[current_pos], regen_cost)
                    else:
                        total_value += self.grid[current_pos]
                        collected_values.add(current_pos)
                moves_left -= 1
    
        if current_pos != self.end_point:
            total_value = 0  # Penalize if the path doesn't end at the end_point
    
        return total_value, path

    def crossover(self, c1, c2):
        """
        Perform crossover between two chromosomes.

        Args:
            c1 (list of tuple of int): First chromosome.
            c2 (list of tuple of int): Second chromosome.

        Returns:
            list of tuple of int: Offspring chromosome after crossover.
        """
        assert len(c1) == len(c2)
        point = random.randrange(len(c1))
        first = random.randrange(2)
        if first:
            return c1[:point] + c2[point:]
        else:
            return c2[:point] + c1[point:]

    def mutate(self, chrom):
        """
        Mutate a chromosome based on mutation parameters.

        Args:
            chrom (list of tuple of int): Chromosome to mutate.

        Returns:
            list of tuple of int: Mutated chromosome.
        """
        return [random.choice(self.moves) if random.randrange(self.mchance) == 0 else move for move in chrom]

    def run_algorithm(self):
        """
        Run the genetic algorithm to find the best path.

        Returns:
            list of tuple of int: The best path found.
            int: The fitness value of the best path.
        """

        tic = time.time()
        
        # Generate initial random population
        pop = []
        for i in range(self.popsize + self.elitismn):
            chrom = [random.choice(self.moves) for _ in range(self.tmax)]
            fit_val, _ = self.fitness(chrom)
            chrom = (fit_val, chrom)
            j = 0
            while i - j > 0 and j < self.elitismn and chrom > pop[i - 1 - j]:
                j += 1
            pop.insert(i - j, chrom)

        bestfit = 0
        for i in range(self.genlimit):
            nextgen = []
            for j in range(self.popsize):
                # Select parents in k tournaments
                parents = sorted(random.sample(pop, self.kt), key=lambda x: x[0])[self.kt - 2:]
                # Crossover and mutate
                offspring = self.mutate(self.crossover(parents[0][1], parents[1][1]))
                fit_val, _ = self.fitness(offspring)
                offspring = (fit_val, offspring)
                if offspring[0] > bestfit:
                    bestfit = offspring[0]
                if self.elitismn > 0 and offspring > pop[self.popsize]:
                    l = 0
                    while l < self.elitismn and offspring > pop[self.popsize + l]:
                        l += 1
                    pop.insert(self.popsize + l, offspring)
                    nextgen.append(pop.pop(self.popsize))
                else:
                    nextgen.append(offspring)
            pop = nextgen + pop[self.popsize:]

        bestchrom = sorted(pop)[self.popsize + self.elitismn - 1]
        bestfit_val, best_path = self.fitness(bestchrom[1])

        tac = time.time()
        elapsed_time_ms = (tac - tic) * 1000.0

        return best_path, bestfit_val, elapsed_time_ms

