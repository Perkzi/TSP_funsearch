from typing import Any

from numpy import floating

specification = r'''
import numpy as np

def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    """Evaluates the total distance of a given TSP route."""
    total_distance = sum(
        distances[route[i], route[i + 1]] for i in range(len(route) - 1)
    )
    total_distance += distances[route[-1], route[0]]  # Return to starting city
    return total_distance
    
def tsp_solve(distances: np.ndarray, start_city: int, num_cities: int) -> list[int]:
    """Generates a route based on the priority function."""
    visited = [start_city]
    current_city = start_city
    for _ in range(num_cities - 1):
        unvisited = [i for i in range(num_cities) if i not in visited]
        next_city = tsp_priority(current_city, unvisited, distances)
        visited.append(next_city)
        current_city = next_city
    return visited

@funsearch.evolve
def tsp_priority(current_city: int, unvisited: list[int], distances: np.ndarray) -> int:
    """Determines the priority of each unvisited city based on the distances."""
    priorities = -distances[current_city, unvisited]  # Smaller distance = higher priority
    next_city_index = np.argmax(priorities)
    return unvisited[next_city_index]


@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of TSP instances."""
    total_scores = []
    for name, instance in instances.items():
        distances = instance['distances']
        num_cities = distances.shape[0]
        route = tsp_solve(distances, start_city=0, num_cities=num_cities)
        total_scores.append(tsp_evaluate(route, distances))
    return -np.mean(total_scores)  # Negative score for minimization
'''

exit()
import numpy as np


def tsp_evaluate(route: list[int], distances: np.ndarray) -> float:
    """Evaluates the total distance of a given TSP route."""
    total_distance = sum(
        distances[route[i], route[i + 1]] for i in range(len(route) - 1)
    )
    total_distance += distances[route[-1], route[0]]  # Return to starting city
    return total_distance


def tsp_solve(distances: np.ndarray, start_city: int, num_cities: int) -> list[int]:
    """Generates a route based on the priority function."""
    visited = [start_city]
    current_city = start_city
    for _ in range(num_cities - 1):
        unvisited = [i for i in range(num_cities) if i not in visited]
        next_city = tsp_priority(current_city, unvisited, distances)
        visited.append(next_city)
        current_city = next_city
    return visited


@funsearch.evolve
def tsp_priority(current_city: int, unvisited: list[int], distances: np.ndarray) -> int:
    """Determines the priority of each unvisited city based on the distances."""
    priorities = -distances[current_city, unvisited]  # Smaller distance = higher priority
    next_city_index = np.argmax(priorities)
    return unvisited[next_city_index]


@funsearch.run
def evaluate(instances: dict) -> floating[Any]:
    """Evaluate heuristic function on a set of TSP instances."""
    total_scores = []
    for name, instance in instances.items():
        distances = instance['distances']
        num_cities = distances.shape[0]
        route = tsp_solve(distances, start_city=0, num_cities=num_cities)
        total_scores.append(tsp_evaluate(route, distances))
    return -np.mean(total_scores)  # Negative score for minimization
