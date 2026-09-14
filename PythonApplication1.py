import time
import torch
import math
import random
#This project is for my own learning of ML(machine learning) and I'll do it from the scratch,
#so torch is just for the tensor operations and math is for some mathematical operations.

class NeuralNetwork:
    def __init__(self, input_size, network_size):
        self.weights = [];
        self.biases = [];
        self.outputs = [];
        self.delta_weights = [];
        self.delta_biases = [];
        self.delta_outputs = [];
        j = input_size;
        for i in network_size:
            temp0 = [[random.random() for k in range(0, j)] for l in range(0, i)];
            temp1 = [[0 for k in range(0, j)] for l in range(0,i)];
            self.weights.append(temp0);
            self.delta_weights.append(temp1);
            j = i;
            self.biases.append([random.random() for k in range(0, i)]);
            self.delta_biases.append([0 for k in range(0, i)]);
            self.outputs.append([0 for k in range(0, i)]);
            self.delta_outputs.append([0 for k in range(0, i)]);
    def return_parameters(self):
        return (self.weights, self.biases, self.outputs)

nn = NeuralNetwork(10, [2,3,4,5,6,1])
print(nn.return_parameters())


            