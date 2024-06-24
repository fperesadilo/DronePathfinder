import unittest
import numpy as np
import random
from orienteering import DynamicProgramming, GeneticAlgorithm 

class TestDynamicProgramming(unittest.TestCase):
    """
    Unit tests for the DynamicProgramming class.
    """

    def test_path_length_matches_T(self):
        """
        Test that the algorithm returns a path of length equal to T.
        """
        grid = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        dp = DynamicProgramming(grid, (0, 0), T=5, regen_value=0)
        path_length = len(dp.max_path_value_and_path()[1])
        self.assertEqual(path_length, 5)

    def test_positive_collected_value(self):
        """
        Test that the maximum path value returned is positive.
        """
        grid = [[-1, -2, -3],
                [-4, -5, -6],
                [-7, -8, -9]]
        dp = DynamicProgramming(grid, (0, 0), T=5, regen_value=0)
        max_value = dp.max_path_value_and_path()[0]
        self.assertGreater(max_value, 0)

    def test_start_point_equals_end_point(self):
        """
        Test that the starting point and ending point of the path are the same.
        """
        grid = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        start = (1, 1)
        dp = DynamicProgramming(grid, start, T=5, regen_value=0)
        path = dp.max_path_value_and_path()[1]
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], start)

    def test_invalid_start_point(self):
        """
        Test handling of invalid starting points outside the grid boundaries.
        """
        grid = [[1, 2],
                [3, 4]]
        start = (2, 2)  # Outside grid boundaries
        dp = DynamicProgramming(grid, start, T=5, regen_value=0)
        with self.assertRaises(IndexError):
            dp.max_path_value_and_path()

class TestGeneticAlgorithm(unittest.TestCase):
    """
    Unit tests for the GeneticAlgorithm class.
    """

    def test_genetic_algorithm_path_length_matches_tmax(self):
        """
        Test that the genetic algorithm returns a path of length equal to tmax.
        """
        grid = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        ga = GeneticAlgorithm(grid, (0, 0), (2, 2), tmax=5, regen_value=0)
        path_length = len(ga.run_algorithm()[0])
        self.assertEqual(path_length, 5)

    def test_genetic_algorithm_positive_collected_value(self):
        """
        Test that the maximum fitness value returned by the genetic algorithm is positive.
        """
        grid = np.array([[-1, -2, -3],
                         [-4, -5, -6],
                         [-7, -8, -9]])
        ga = GeneticAlgorithm(grid, (0, 0), (2, 2), tmax=5, regen_value=0)
        fitness_value = ga.run_algorithm()[1]
        self.assertGreater(fitness_value, 0)

    def test_genetic_algorithm_start_point_equals_end_point(self):
        """
        Test that the starting point and ending point of the path found by the genetic algorithm are the same.
        """
        grid = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        start = (1, 1)
        ga = GeneticAlgorithm(grid, start, start, tmax=5, regen_value=0)
        path = ga.run_algorithm()[0]
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], start)

    def test_genetic_algorithm_small_population_and_generation_limit(self):
        """
        Test the genetic algorithm with a small population size and generation limit.
        """
        grid = np.array([[1, 2],
                         [3, 4]])
        ga = GeneticAlgorithm(grid, (0, 0), (1, 1), tmax=5, regen_value=0, popsize=5, genlimit=10)
        path = ga.run_algorithm()[0]
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)

    def test_genetic_algorithm_mutation_probability(self):
        """
        Test the mutation probability in the genetic algorithm.
        """
        grid = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])
        ga = GeneticAlgorithm(grid, (0, 0), (2, 2), tmax=5, regen_value=0, mchance=1)
        initial_path = ga.run_algorithm()[0]
        mutated_paths = []
        for _ in range(5):
            mutated_paths.append(ga.run_algorithm()[0])
        for path in mutated_paths:
            self.assertNotEqual(path, initial_path)

if __name__ == '__main__':
    unittest.main()

