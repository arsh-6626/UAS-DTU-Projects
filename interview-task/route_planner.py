import numpy as np
import cv2 as cv

def calculate_distance(city1, city2):
    # Calculate the Euclidean distance between two triangle centroids by distance formula
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def nearest_neighbor_algorithm(cities):
    n = len(cities)
    visited = [False] * n  # Track visited cities
    tour = []  # This will store the order of visited cities
    current_city = 0  # Start from the first city
    visited[current_city] = True
    tour.append(current_city)

    for i in range(n - 1):  # Loop through all cities except the starting one
        nearest_city = None
        nearest_distance = float('inf')

        for city_index in range(n):
            if not visited[city_index]:  # If the city hasn't been visited
                distance = calculate_distance(cities[current_city], cities[city_index])
                if distance < nearest_distance:  # Check if this is the nearest city
                    nearest_distance = distance
                    nearest_city = city_index
        
        # Move to the nearest city
        visited[nearest_city] = True
        tour.append(nearest_city)
        current_city = nearest_city

    # Return to the starting city to complete the cycle
    tour.append(0)  # create a complete cycle
    return tour

def calculate_total_distance(tour, cities):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += calculate_distance(cities[tour[i]], cities[tour[i + 1]])
    return total_distance
