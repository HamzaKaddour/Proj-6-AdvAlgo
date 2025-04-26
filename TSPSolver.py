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

class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None
        self._bssf = None
        self._time_limit = 60.0

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            perm = np.random.permutation(ncities)
            route = [cities[i] for i in perm]
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def greedy(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        best_tour = None
        best_cost = np.inf
        start_time = time.time()

        for start in range(ncities):
            unvisited = set(range(ncities))
            route = [start]
            unvisited.remove(start)
            current_city = start

            while unvisited:
                next_city = min(unvisited, key=lambda i: cities[current_city].costTo(cities[i]))
                if cities[current_city].costTo(cities[next_city]) == np.inf:
                    break
                route.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city

            if len(route) == ncities:
                tour = TSPSolution([cities[i] for i in route])
                if tour.cost < best_cost:
                    best_cost = tour.cost
                    best_tour = tour

            if time.time() - start_time > time_allowance:
                break

        return {'cost': best_cost, 'soln': best_tour} if best_tour else self.defaultRandomTour(time_allowance)

    def reduce_matrix(self, matrix):
        reduction_cost = 0
        size = len(matrix)
        for i in range(size):
            row = matrix[i]
            min_val = min(row)
            if min_val != np.inf and min_val > 0:
                reduction_cost += min_val
                for j in range(size):
                    if matrix[i][j] != np.inf:
                        matrix[i][j] -= min_val
        for j in range(size):
            col = [matrix[i][j] for i in range(size)]
            min_val = min(col)
            if min_val != np.inf and min_val > 0:
                reduction_cost += min_val
                for i in range(size):
                    if matrix[i][j] != np.inf:
                        matrix[i][j] -= min_val
        return matrix, reduction_cost

    def copy_matrix(self, matrix):
        return [row[:] for row in matrix]

    def branchAndBound(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        start_time = time.time()
        self._time_limit = time_allowance
        bssf_time = None

        bssf_solution = self.greedy(time_allowance / 10)
        bssf = bssf_solution.get('soln', None)
        bssf_cost = bssf.cost if bssf else np.inf
        self._bssf = bssf

        if bssf:
            bssf_time = time.time()
            solutions = 1
        else:
            solutions = 0

        matrix = [[cities[i].costTo(cities[j]) for j in range(ncities)] for i in range(ncities)]
        matrix, cost = self.reduce_matrix(matrix)

        pq = []
        state_id = itertools.count()
        heapq.heappush(pq, (cost, 0, next(state_id), [0], matrix, cost))

        total_states = 1
        pruned_states = 0
        max_q_size = 1

        while pq and time.time() - start_time < self._time_limit:
            bound, depth, _, path, matrix, path_cost = heapq.heappop(pq)
            current_city = path[-1]

            if bound >= bssf_cost:
                pruned_states += 1
                continue

            if len(path) == ncities:
                final_cost = path_cost + cities[current_city].costTo(cities[path[0]])
                if final_cost < bssf_cost:
                    bssf = TSPSolution([cities[i] for i in path])
                    bssf_cost = bssf.cost
                    self._bssf = bssf
                    bssf_time = time.time()
                    solutions += 1
                continue

            for next_city in range(ncities):
                if next_city in path:
                    continue
                new_matrix = self.copy_matrix(matrix)
                for j in range(ncities):
                    new_matrix[current_city][j] = np.inf
                for i in range(ncities):
                    new_matrix[i][next_city] = np.inf
                new_matrix[next_city][path[0]] = np.inf

                step_cost = matrix[current_city][next_city]
                reduced_matrix, reduction = self.reduce_matrix(new_matrix)
                new_bound = path_cost + step_cost + reduction

                if new_bound >= bssf_cost:
                    pruned_states += 1
                    continue

                new_path = path + [next_city]
                heapq.heappush(pq, (new_bound, len(new_path), next(state_id), new_path, reduced_matrix, path_cost + step_cost))
                total_states += 1
                max_q_size = max(max_q_size, len(pq))

        end_time = time.time()
        solved_time = (bssf_time - start_time) if bssf_time else (end_time - start_time)

        return {
            'cost': bssf_cost,
            'time': solved_time,
            'count': solutions,
            'soln': bssf,
            'max': max_q_size,
            'total': total_states,
            'pruned': pruned_states
        }




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