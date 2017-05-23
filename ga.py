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


class GA:
    """ Manages a genetic algorithm to find optimal weights """
    population_size = 0
    population = []
    lvq = None

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
        for _ in range(num_iterations):
            print("iterate")
            self.population.sort(key=lambda x: self.lvq.average_inter_cluster_distance(x))
            for index in range(self.population_size):
                self.population[index] = self.mutate(self.population[index])

    def select(self):
        """ yes """
        pass

    def mutate(self, weight_matrix):
        """ Performs a random mutation on an individual an returns the result """
        for row in weight_matrix:
            for index in range(len(row)):
                row[index] = row[index] #+ random.random()
        return weight_matrix

    def cross_over(self, first_parent, second_parent):
        """ Makes a new weight matrix from two older ones """
        pass

    def get_best_individual(self):
        """ Returns the most fit individual in the population currently """
        self.population.sort(key=lambda x: self.lvq.average_inter_cluster_distance(x))
        return self.population[0]
