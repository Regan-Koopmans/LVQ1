#!/bin/python3

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

Module      : Main
Description : Invokes the module LVQ to
              train a neural network using
              information passed through
              the command line.

"""

from lvq import LVQ

def main():
    """The main entry point to the program"""
    lvq = LVQ()
    lvq.parse_args()
    lvq.parse_data_file()
    lvq.initialize_weights()
    lvq.train()
    if lvq.dynamic_clusters:
        print("Silhouette %f " % lvq.silhouette())
        silhouette_index = lvq.silhouette()
        while silhouette_index < 0.5:
            lvq.args.clusters += 1
            lvq.initialize_weights()
            lvq.train()
            silhouette_index = lvq.silhouette()
    write_results(lvq)


def write_results(lvq):
    """ Writes the appropriate statistics to the terminal and a file """
    file = open(lvq.args.output, "w")
    output(file, "")
    for index in range(len(lvq.weight_matrix)):
        output(file, "Centroid " + str(index+1) + ": "  + array_to_string(lvq.weight_matrix[index]))
    output(file, "")
    lvq.allocate_clusters(lvq.weight_matrix, lvq.training_set)
    lvq.allocate_clusters(lvq.weight_matrix, lvq.testing_set)
    output(file, "Quantization error (on training): %f" % lvq.average_intra_cluster_distance_all_clusters(lvq.weight_matrix, lvq.training_set))
    output(file, "Quantization error (on testing): %f" % lvq.average_intra_cluster_distance_all_clusters(lvq.weight_matrix, lvq.testing_set))
    output(file, "Average inter-cluster distance: %f" % lvq.average_inter_cluster_distance(lvq.weight_matrix))
    output(file, "")
    for index in range(lvq.args.clusters):
        output(file, "Intra-distance cluster " + str(index+1) + ": %f " % lvq.average_intra_cluster_distance(lvq.weight_matrix, index, lvq.training_set))
    output(file, "")
    output(file, "Clustering on unseen (testing) data:\n")
    for index in range(len(lvq.testing_set)):
        output(file, "Pattern " + str(index+1) + ": " + str(lvq.testing_set[index][-1]))
    file.close()

def output(file, msg):
    """ Prints a message to the screen and writes it to the output file. """
    print(msg)
    file.write(msg + "\n")

def array_to_string(array):
    """Creates a formatted string from an array of elements"""
    string = ""
    for element in array:
        string += "%f " % element
    return string

if __name__ == "__main__":
    main()
