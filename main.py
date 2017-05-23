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
    file = open(lvq.args.output, "w")
    output(file, "")
    for index in range(len(lvq.weight_matrix)):
        output(file, "Centroid " + str(index+1) + ": "  + array_to_string(lvq.weight_matrix[index]))
    output(file, "")
    lvq.allocate_clusters()
    output(file, "Quantization error: " +
           str(lvq.average_inter_cluster_distance(lvq.weight_matrix)))
    output(file, "Average inter-cluster distance: " +
           str(lvq.average_inter_cluster_distance(lvq.weight_matrix)))
    output(file, "")
    for index in range(lvq.args.clusters):
        output(file, "Intra-distance cluster " + str(index+1) + ": " +
               str(lvq.average_intra_cluster_distance(lvq.weight_matrix, index)))
    output(file, "")
    for index in range(len(lvq.data_set)):
        output(file, "Pattern " + str(index+1) + ": " + str(lvq.data_set[index][-1]))
    file.close()

def output(file, msg):
    """ Prints a message to the screen and writes it to the output file. """
    #print(msg)
    file.write(msg + "\n")

def array_to_string(array):
    """Creates a formatted string from an array of elements"""
    string = ""
    for element in array:
        string += str(element) + " "
    return string

if __name__ == "__main__":
    main()
