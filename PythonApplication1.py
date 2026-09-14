import time
from turtle import forward
#import torch
import math
import random
#This project is for my own learning of ML(machine learning) and I'll do it from the scratch without torch

def sigmoid(x):
    return 1/(1 + math.exp(-x))
def sigmoid_derivative(x):
    return x * (1 -x )

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
            temp0 = [[random.uniform(-0.5, 0.5) for k in range(0, j)] for l in range(0, i)];
            temp1 = [[0 for k in range(0, j)] for l in range(0,i)];
            self.weights.append(temp0);
            self.delta_weights.append(temp1);
            j = i;
            self.biases.append([random.uniform(-0.5, 0.5) for k in range(0, i)]);
            self.delta_biases.append([0 for k in range(0, i)]);
            self.outputs.append([0 for k in range(0, i)]);
            self.delta_outputs.append([0 for k in range(0, i)]);
    def return_parameters(self):
        return (self.weights, self.biases, self.outputs)
    def forward(self, inputs):
        self.outputs[0] = [sigmoid(sum([inputs[k] * self.weights[0][j][k] for k in range(0, len(inputs))]) + self.biases[0][j]) for j in range(0, len(self.weights[0]))];
        for i in range(1, len(self.weights)):
            self.outputs[i] = [sigmoid(sum([self.outputs[i-1][k] * self.weights[i][j][k] for k in range(0, len(self.outputs[i-1]))]) + self.biases[i][j]) for j in range(0, len(self.weights[i]))];
        return self.outputs[-1];
    def backward(self, inputs, target, learning_rate):
        L = len(self.outputs)
        last = L - 1

        self.delta_outputs[last] = [
            (self.outputs[last][k] - target[k]) * sigmoid_derivative(self.outputs[last][k])
            for k in range(len(self.outputs[last]))
        ]

        for i in range(L - 2, -1, -1):
            next_size = len(self.outputs[i + 1])
            cur_size = len(self.outputs[i])
            self.delta_outputs[i] = [
                sigmoid_derivative(self.outputs[i][k]) *
                sum(self.delta_outputs[i + 1][l] * self.weights[i + 1][l][k]
                    for l in range(next_size))
                for k in range(cur_size)
            ]

        for i in range(L):
            layer_input = inputs if i == 0 else self.outputs[i - 1]
            for j in range(len(self.outputs[i])):
                for k in range(len(layer_input)):
                    self.weights[i][j][k] -= learning_rate * self.delta_outputs[i][j] * layer_input[k]
                self.biases[i][j] -= learning_rate * self.delta_outputs[i][j]

nn = NeuralNetwork(2, [2,3,3,1])
for i in range(10000):
    inputs = [random.random() for i in range(0,2)]
    targets = [inputs[0] * inputs[1] * 3 ]
    nn.forward(inputs)
    nn.backward(inputs, targets, 0.01)
print(nn.forward([0.5, 0.2]))
#print(nn.return_parameters())