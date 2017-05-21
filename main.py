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
    print()
    for index in range(len(lvq.weight_matrix)):
        print("Centroid " + str(index+1) + ": "  + array_to_string(lvq.weight_matrix[index]))
    print()
    lvq.allocate_clusters()
    print("Quantization error: " + str(lvq.average_inter_cluster_distance()))
    print("Average inter-cluster distance: " + str(lvq.average_inter_cluster_distance()))
    print()
    for index in range(lvq.args.clusters):
        print("Intra-distance cluster " + str(index+1) + ": " +  str(lvq.average_intra_cluster_distance(index)))
    print()
    for index in range(len(lvq.data_set)):
        print("Pattern " + str(index+1) + ": " + str(lvq.data_set[index][-1]))

def array_to_string(array):
    """Creates a formatted string from an array of elements"""
    output = ""
    for element in array:
        output += str(element) + " "
    return output

if __name__ == "__main__":
    main()
