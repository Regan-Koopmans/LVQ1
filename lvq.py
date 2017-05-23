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

Module      : LVQ
Description : Encapsulates all functionality
              needed to train a simple LVQ
              neural network.
"""

from random import uniform
from argparse import ArgumentParser
import math
from ga import GA

def fatal_error(msg):
    """ Prints out an error message and exits """
    print()
    print("Error: " + msg)
    print()
    exit()

class LVQ:
    """ Manages the training of an LVQ1 neural network """
    args = {}
    data_set = []
    learning_rate = 1
    pattern_length = 0
    weight_matrix = []

    def parse_args(self):
        """ Parses the command line arguments """
        parser = ArgumentParser()
        parser.add_argument("data", help="The input set to cluster from.")
        parser.add_argument("clusters", help="The number of clusters to create.", type=int)
        parser.add_argument("output", help="The file to write the output to.")
        parser.add_argument("test", help="The file to test the centroid vectors on.")
        parser.add_argument("algorithm", help="The algorithm to use.", type=int)
        parser.add_argument("iterations", help="The number of iterations.", type=int)
        self.args = parser.parse_args()

    def add_entry(self, line):
        """ Adds a single data entry by processing a single line from the file. """
        line = line.strip()
        if line != "":
            self.data_set.append(list(map(lambda x: float(x), line.split("\t"))))

    def parse_data_file(self):
        """ Reads data from filename passed throught argument """
        file = open(self.args.data)
        for line in file:
            self.add_entry(line)
        file.close()
        self.pattern_length = len(self.data_set[0])
        for pattern in self.data_set:
            if len(pattern) != self.pattern_length:
                fatal_error("Mismatch between pattern lengths in data set.")

    def get_maxima_array(self):
        """Retrieves the min and max for each attribute in the data set """
        maxima_array = [{"maximum" : None, "minimum" :  None} for x in range(self.pattern_length)]
        for entry in self.data_set:
            for index in range(self.pattern_length):
                if maxima_array[index]["maximum"] is None or \
                entry[index] > maxima_array[index]["maximum"]:
                    maxima_array[index]["maximum"] = entry[index]
                if maxima_array[index]["minimum"] is None or \
                entry[index] < maxima_array[index]["minimum"]:
                    maxima_array[index]["minimum"] = entry[index]
        return maxima_array

    def initialize_weights(self):
        """  Initializes the network weights to a value
        uniformly distributed between min and max """
        maxima_array = self.get_maxima_array()
        print("Initializing matrix with " + str(self.args.clusters) + " x " +
              str(self.pattern_length))
        self.weight_matrix = [[uniform(maxima_array[x]["minimum"], maxima_array[x]["maximum"]) \
                            for x in range(self.pattern_length)] for y in range(self.args.clusters)]

    def euclid_dist(self, from_vector, to_vector):
        """ Calculates the Euclidean distance between two vectors of similar dimension. """
        total = 0
        for index in range(self.pattern_length):
            total += math.pow(from_vector[index] - to_vector[index], 2)
        return math.sqrt(total)

    def coerce_vector(self, input_vector, to_vector):
        """ Will move a vector to be closer to another, in proportion to the learning rate """
        for index in range(self.pattern_length):
            input_vector[index] = input_vector[index] - \
                self.learning_rate*(input_vector[index] - to_vector[index])
        self.learning_rate *= 0.95
        return input_vector

    def train(self):
        """ Trains the neural network either competitively or using a GA """
        if self.args.algorithm == 0:
            for _ in range(self.args.iterations):
                for entry in self.data_set:
                    smallest_distance = float('inf')
                    smallest_distance_index = -1
                    for index in range(len(self.weight_matrix)):
                        current_dist = self.euclid_dist(entry, self.weight_matrix[index])
                        if current_dist < smallest_distance:
                            smallest_distance = current_dist
                            smallest_distance_index = index
                    self.weight_matrix[smallest_distance_index] = \
                      self.coerce_vector(self.weight_matrix[smallest_distance_index], entry)
        else:
            genetic_algorithm = GA(25, self)
            genetic_algorithm.initialize_population()
            genetic_algorithm.iterate(self.args.iterations)
            self.weight_matrix = genetic_algorithm.get_best_individual()

    def allocate_clusters(self, weight_matrix):
        """ Gives a cluster to every data entry """
        for entry in self.data_set:
            min_cluster = None
            min_cluster_distance = None
            for index in range(len(weight_matrix)):
                if min_cluster is None or \
                    self.euclid_dist(entry, weight_matrix[index]) < min_cluster_distance:
                    min_cluster = index
                    min_cluster_distance = self.euclid_dist(entry, weight_matrix[index])
            if len(entry) == self.pattern_length:
                entry.append(min_cluster)
            else:
                entry[self.pattern_length] = min_cluster

    def average_inter_cluster_distance(self, weight_matrix):
        """ Calculates average intercluster distance """
        total = 0
        for from_vector in weight_matrix:
            for to_vector in weight_matrix:
                total += self.euclid_dist(from_vector, to_vector)
        return total / len(weight_matrix)**2

    def average_intra_cluster_distance_all_clusters(self, weight_matrix):
        """ asdf """
        total = 0
        for cluster in range(self.args.clusters):
            total += self.average_intra_cluster_distance(weight_matrix, cluster);
        return total / self.args.clusters

    def average_intra_cluster_distance(self, weight_matrix, cluster):
        """ Calculates average intracluster distance for a centroid """
        total = 0
        count = 0
        for entry in self.data_set:
            if entry[-1] == cluster:
                count += 1
                total += self.euclid_dist(entry, weight_matrix[cluster])
        if count != 0:
            return total / count
        return 0
