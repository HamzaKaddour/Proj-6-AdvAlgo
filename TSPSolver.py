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
from random import randint

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


    INT_MAX = 2147483647
    numberOfCities = 5
    populationSize = 10
    # genetic algorithm
    def fancy(self, time_allowance=60.0):
        solutions_found = 0
        total_individuals = 0

        citys = self._scenario.getCities()
        self.numberOfCities = len(citys)

        generationNum = 1
        #  update this number to increase the number of generations and get a better solution
        geneIterations = 50
        population = []
        
        # Populate the gnome pool
        for i in range(self.populationSize):
            temp = individual()
            temp.gnome = self.create_gnome()
            temp.fitness = self.calculate_fitness(temp.gnome)
            population.append(temp)
        
        found = False
        temperature = 10000

        start_time = time.time()
        while generationNum <= geneIterations and time.time() - start_time < time_allowance:
            population.sort()
            newPopulation = []

            for i in range(self.populationSize):
                p1 = population[i]

                while True:
                    newG = self.mutatedGene(p1.gnome)
                    newGnome = individual()
                    newGnome.gnome = newG
                    newGnome.fitness = self.calculate_fitness(newGnome.gnome)
                    total_individuals += 1

                    if newGnome.fitness < population[i].fitness:
                        newPopulation.append(newGnome)
                        solutions_found += 1  # found a better solution
                        break
                    else:
                        prob = pow(2.7, -1 * (float(newGnome.fitness - population[i].fitness) / temperature))
                        if prob > 0.5:
                            newPopulation.append(newGnome)
                            break

            temperature = self.cooldown(temperature)
            population = newPopulation
            generationNum += 1

        # Pick the best individual
        best = min(population, key=lambda ind: ind.fitness)

        route = [citys[idx] for idx in best.gnome[:-1]]

        # create TSPSolution
        bssf = TSPSolution(route)

        end_time = time.time()

        results = {}
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = solutions_found
        results['soln'] = bssf
        results['max'] = self.populationSize
        results['total'] = total_individuals
        results['pruned'] = None # you do not prune in genetic algorithm

        return results



    # helper functions
    def mutatedGene(self, gnome):
        gnome = gnome.copy()
        while True:
            r = self.rand_num(1, self.numberOfCities) - 1
            r1 = self.rand_num(1, self.numberOfCities) - 1
            if r != r1:
                gnome[r], gnome[r1] = gnome[r1], gnome[r]
                break
        return gnome



    def rand_num(self, start, end):
        return randint(start, end-1)
    
    def create_gnome(self):
        gnome = [0]
        available = list(range(1, self.numberOfCities))
        while available:
            next_city = available.pop(randint(0, len(available)-1))
            gnome.append(next_city)
        gnome.append(0)  # to make it a complete tour
        return gnome


    # Function to check if the character has already occurred in the string
    def repeat(self, s, ch):
        for i in range(len(s)):
            if s[i] == ch:
                return True

        return False
    
    # Function to return the updated value of the cooling element.
    def cooldown(self, temp):
        return (90 * temp) / 100

    # Function to return the fitness value of a gnome. The fitness value is the path length of the path represented by the GNOME.
    def calculate_fitness(self, gnome):
        cities = self._scenario.getCities()
        f = 0
        for i in range(len(gnome) - 1):
            c1_idx = gnome[i]
            c2_idx = gnome[i+1]
            cost = cities[c1_idx].costTo(cities[c2_idx])
            if cost == np.inf:
                return self.INT_MAX
            f += cost
        return f



class individual:
    def __init__(self) -> None:
        self.gnome = ""
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness
