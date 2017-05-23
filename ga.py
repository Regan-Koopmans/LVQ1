"""
########################################
#      University of Pretoria          #
#   Department of Computer Science     #
#  COS 314 - Artificial Intelligence   #
#           Regan Koopmans             #
#                                      #
#     Project 3 - Neural Networks      #
#         and Machine Learning         #
########################################

Module      : GA
Description : Encapsulates a genetic algorithm
              for determining optimal weights.

"""

import random
from random import uniform
from copy import deepcopy

class GA:
    """ Manages a genetic algorithm to find optimal weights """
    population_size = 0
    population = []
    mutate_rate = 0.01;
    lvq = None
    all_time_best = None

    def __init__(self, population_size, lvq):
        self.population_size = population_size
        self.lvq = lvq

    def initialize_population(self):
        """ Initializes the population to random weight matrices """
        maxima_array = self.lvq.get_maxima_array()
        for _ in range(self.population_size):
            self.population.append([[uniform(maxima_array[x]["minimum"], \
                                    maxima_array[x]["maximum"]) \
                                    for x in range(self.lvq.pattern_length)]\
                                    for y in range(self.lvq.args.clusters)])

    def iterate(self, num_iterations):
        """ Itererates num_iterations number of times """
        for counter in range(num_iterations):
            best = self.get_best_individual_index()
            for index in range(self.population_size):
                if index != 0:
                    self.population.append(self.mutate(self.cross_over(self.population[index], self.population[best]), index))
            self.population.sort(key=self.fitness)
            self.population = self.population[0:self.population_size]
            print(str(self.population[0]))
            self.mutate_rate = self.mutate_rate

    def fitness(self, matrix):
        """ Calculates and returns the fitness of a weight matrix """
        self.lvq.allocate_clusters(matrix)
        return self.lvq.average_intra_cluster_distance_all_clusters(matrix) - self.lvq.average_inter_cluster_distance(matrix)

    def select(self):
        """ yes """
        pass

    def mutate(self, weight_matrix, index):
        """ Performs a random mutation on an individual an returns the result """
        for row in weight_matrix:
            for index in range(len(row)):
                row[index] += self.mutate_rate*(random.uniform(-1, 1))
        return weight_matrix

    def cross_over(self, first_parent, second_parent):
        """ Makes a new weight matrix from two older ones """
        # child_1 = deepcopy(first_parent)
        # child_2 = deepcopy(second_parent)
        # for row in range(self.lvq.args.clusters):
        #     for item in range(self.lvq.pattern_length):
        #         if (row + item) % 2 == 0:
        #             child_1[row][item] = second_parent[row][item]
        #             child_2[row][item] = first_parent[row][item]
        # if self.fitness(child_1) > self.fitness(child_2):
        #     return child_1
        # return child_2
        child = deepcopy(second_parent)
        for row in range(self.lvq.args.clusters):
             for item in range(self.lvq.pattern_length):
                child[row][item] = (first_parent[row][item] + second_parent[row][item]) / 2
        return child

    def get_best_individual_index(self):
        max_value = None
        max_index = None
        for index in range(self.population_size):
            current_value = self.fitness(self.population[index])
            if max_value is None or current_value > max_value:
                max_value = current_value
                max_index = index
        return max_index

    def get_best_individual(self):
        """ Returns the most fit individual in the population currently """
        self.population.sort(key=self.fitness)
        return self.population[0]
