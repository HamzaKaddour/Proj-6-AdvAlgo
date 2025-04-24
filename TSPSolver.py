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

    def fancy(self, time_allowance=60.0):
        pass
