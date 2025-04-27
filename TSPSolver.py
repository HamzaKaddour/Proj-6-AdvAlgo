#!/usr/bin/python3

# Determine PyQt version for compatibility
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

# Represents a search state in the Branch-and-Bound tree
class State:
    def __init__(self, path, matrix, cost_so_far, lower_bound):
        self.path = path  # List of visited city indices
        self.matrix = matrix  # Reduced cost matrix
        self.cost_so_far = cost_so_far  # Accumulated cost of taken path
        self.lower_bound = lower_bound  # Cost of further reductions
        self.priority = self.cost_so_far + self.lower_bound  # Used by priority queue

    def __lt__(self, other):  # Enables comparison in heapq
        return self.priority < other.priority


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    # Constructs full cost matrix from all cities (inf on diagonals)
    def build_initial_matrix(self, cities):
        n = len(cities)
        matrix = np.full((n, n), math.inf)
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = cities[i].costTo(cities[j])
        return matrix

    # Reduces the matrix and computes the lower bound
    def reduce_matrix(self, matrix):
        matrix = matrix.copy()
        bound = 0

        # Row reduction step
        for i in range(len(matrix)):
            row = matrix[i]
            finite_vals = row[np.isfinite(row)]
            if len(finite_vals) == 0:
                continue
            min_val = np.min(finite_vals)
            if min_val > 0:
                matrix[i] -= min_val
                bound += min_val

        # Column reduction step
        for j in range(len(matrix[0])):
            col = matrix[:, j]
            finite_vals = col[np.isfinite(col)]
            if len(finite_vals) == 0:
                continue
            min_val = np.min(finite_vals)
            if min_val > 0:
                matrix[:, j] -= min_val
                bound += min_val

        return matrix, bound

    def print_matrix(self, matrix, label="Matrix"):
        print(f"\n{label}:")
        for row in matrix:
            row_str = "  ".join(f"{v:5.0f}" if math.isfinite(v) else "  ---" for v in row)
            print(row_str)

    # Generates a random tour (used for backup BSSF)
    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        while not foundTour and time.time()-start_time < time_allowance:
            perm = np.random.permutation(ncities)
            route = [cities[i] for i in perm]
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                foundTour = True

        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = time.time() - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = results['total'] = results['pruned'] = None
        return results

    # Greedy tour used for initial BSSF, tries to always take shortest legal move
    def greedy(self, time_allowance=60.0):
        start_time = time.time()
        cities = self._scenario.getCities()
        ncities = len(cities)

        best_solution = None
        best_cost = math.inf
        solutions_found = 0

        for start_index in range(ncities):
            visited = set()
            route = []
            current_city = cities[start_index]
            route.append(current_city)
            visited.add(current_city)

            while len(route) < ncities:
                next_city = None
                min_cost = math.inf
                for candidate in cities:
                    if candidate not in visited:
                        cost = current_city.costTo(candidate)
                        if cost < min_cost:
                            min_cost = cost
                            next_city = candidate
                if next_city is None or min_cost == math.inf:
                    break

                route.append(next_city)
                visited.add(next_city)
                current_city = next_city

            if len(route) == ncities and route[-1].costTo(route[0]) != math.inf:
                candidate_solution = TSPSolution(route)
                if candidate_solution.cost < best_cost:
                    best_solution = candidate_solution
                    best_cost = candidate_solution.cost
                    solutions_found += 1

            if time.time() - start_time > time_allowance:
                break

        return {
            'cost': best_solution.cost if best_solution else math.inf,
            'time': time.time() - start_time,
            'count': solutions_found,
            'soln': best_solution,
            'max': None, 'total': None, 'pruned': None
        }

    # Main Branch and Bound TSP algorithm
    def branchAndBound(self, time_allowance=60.0):
        start_time = time.time()
        results = {}
        cities = self._scenario.getCities()
        n = len(cities)

        # Generate initial BSSF using greedy or fallback to random
        bssf = self.greedy(time_allowance=1.0)['soln']
        if bssf is None or bssf.cost == math.inf:
            bssf = self.defaultRandomTour(time_allowance=1.0)['soln']
        best_cost = bssf.cost

        print(f"Initial BSSF cost: {bssf.cost}")

        # Create initial reduced cost matrix and state
        initial_matrix = self.build_initial_matrix(cities)
        self.print_matrix(initial_matrix, "Initial Cost Matrix")

        reduced_matrix, lb = self.reduce_matrix(initial_matrix)
        self.print_matrix(reduced_matrix, "Reduced Matrix")
        print("Initial lower bound:", lb)

        initial_state = State(path=[0], matrix=reduced_matrix, cost_so_far=0, lower_bound=lb)
        pq = [initial_state]
        heapq.heapify(pq)

        solutions_found = 0
        total_states = 1
        pruned_states = 0
        max_queue_size = 1

        while time.time() - start_time < time_allowance and pq:
            current_state = heapq.heappop(pq)
            print(f"Checking state {current_state.path} | priority: {current_state.priority:.2f}, best_cost: {best_cost:.2f}")

            if current_state.priority >= best_cost:
                print(f"PRUNED: path={current_state.path}, priority={current_state.priority}, BSSF={best_cost}")
                pruned_states += 1
                continue

            current_city = current_state.path[-1]

            if len(current_state.path) == n:
                return_cost = cities[current_city].costTo(cities[0])
                if return_cost != math.inf:
                    total_cost = current_state.cost_so_far + return_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        route = [cities[i] for i in current_state.path]
                        bssf = TSPSolution(route)
                        bssf.cost = total_cost
                        solutions_found += 1
                        print(f"New BSSF found! Cost: {total_cost}, Path: {current_state.path}")
                continue

            for next_city in range(n):
                if next_city in current_state.path:
                    continue

                cost_to_next = current_state.matrix[current_city][next_city]
                if cost_to_next == math.inf:
                    continue

                new_path = current_state.path + [next_city]
                new_matrix = current_state.matrix.copy()

                new_matrix[current_city, :] = math.inf
                new_matrix[:, next_city] = math.inf
                new_matrix[next_city, 0] = math.inf  # prevent early return

                reduced, reduction_cost = self.reduce_matrix(new_matrix)
                total_cost_so_far = current_state.cost_so_far + cost_to_next
                new_lb = reduction_cost
                total_lb = total_cost_so_far + new_lb

                if total_lb < best_cost:
                    new_state = State(new_path, reduced, total_cost_so_far, new_lb)
                    heapq.heappush(pq, new_state)
                    total_states += 1
                    max_queue_size = max(max_queue_size, len(pq))
                else:
                    pruned_states += 1

        results['cost'] = bssf.cost
        results['time'] = time.time() - start_time
        results['count'] = solutions_found
        results['soln'] = bssf
        results['max'] = max_queue_size
        results['total'] = total_states
        results['pruned'] = pruned_states

        print("\n=== Branch and Bound Summary ===")
        print(f"Total solutions found: {solutions_found}")
        print(f"Final BSSF cost: {bssf.cost}")
        print(f"Pruned states: {pruned_states}")
        print(f"Total states created: {total_states}")

        return results


    def fancy(self, time_allowance=60.0):
        start_time = time.time()
        cities = self._scenario.getCities()
        ncities = len(cities)

        pop_size = 100
        generations = 200
        mutation_rate = 0.1

        # Create initial population (list of tours, each as list of city indices)
        population = self._initialize_population(cities, pop_size)

        best_tour = None
        best_cost = float('inf')

        for gen in range(generations):
            if time.time() - start_time > time_allowance:
                break

            fitness = self._evaluate_fitness(population, cities)
            new_population = []

            for _ in range(pop_size):
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                child = self._order_crossover(parent1, parent2)
                self._mutate(child, mutation_rate)
                new_population.append(child)

            population = new_population

            for tour in population:
                route = [cities[i] for i in tour]
                sol = TSPSolution(route)
                if sol.cost < best_cost:
                    best_tour = sol
                    best_cost = sol.cost

            # Keep best from previous gen
            best_idx = np.argmax(fitness)
            best_tour = population[best_idx]
            new_population = [best_tour[:]]  # Elitism

        return {
            'cost': best_tour.cost,
            'time': time.time() - start_time,
            'count': 1,
            'soln': best_tour,
            'max': None,
            'total': None,
            'pruned': None
        }

    #Helper Functions for GA
    def _initialize_population(self, cities, pop_size):
        ncities = len(cities)
        population = []

        # Add 10% greedy tours
        greedy_sol = self.greedy(time_allowance=1.0)['soln']
        if greedy_sol:
            greedy_tour = [cities.index(city) for city in greedy_sol.route]
            for _ in range(int(pop_size * 0.1)):
                population.append(greedy_tour[:])

        # Fill the rest randomly
        for _ in range(pop_size - len(population)):
            tour = list(np.random.permutation(ncities))
            population.append(tour)
        return population

    def _evaluate_fitness(self, population, cities):
        fitness = []
        for tour in population:
            route = [cities[i] for i in tour]
            sol = TSPSolution(route)
            cost = sol.cost
            fitness.append(1 / cost if cost != 0 else 0)
        return fitness

    def _tournament_selection(self, population, fitness, k=3):
        selected = np.random.choice(len(population), k, replace=False)
        best = selected[0]
        for i in selected[1:]:
            if fitness[i] > fitness[best]:
                best = i
        return population[best][:]

    def _order_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size

        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        child[start:end] = parent1[start:end]

        p2_index = end
        c_index = end
        while None in child:
            if parent2[p2_index % size] not in child:
                child[c_index % size] = parent2[p2_index % size]
                c_index += 1
            p2_index += 1

        return child

    def _mutate(self, tour, mutation_rate):
        if np.random.rand() < mutation_rate:
            i, j = np.random.choice(len(tour), 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]


'''
# Simple test for 4-city square
if __name__ == "__main__":
    from PyQt5.QtCore import QPointF

    city_points = [QPointF(0, 0), QPointF(1, 0), QPointF(1, 1), QPointF(0, 1)]
    scenario = Scenario(city_points, difficulty="Easy", rand_seed=42)

    solver = TSPSolver(None)
    solver.setupWithScenario(scenario)
    print("Running Branch and Bound on 4-city square...\n")
    results = solver.branchAndBound(time_allowance=5.0)

    print("====== Results ======")
    print(f"Cost: {results['cost']}")
    print(f"Time: {results['time']:.4f} seconds")
    print(f"Solutions found: {results['count']}")
    print(f"Max queue size: {results['max']}")
    print(f"Total states created: {results['total']}")
    print(f"States pruned: {results['pruned']}")
    print(f"Tour: {[city._name for city in results['soln'].route]}")
'''