import math
import random
from turtle import forward
#import torch 
#=========no torch is allowed in this project=========#

def acti_func(x, func_type):
    if func_type == 0:
        return 1 / (1 + math.exp(-x))
    if func_type == 1:
        return max(0, x)
    if func_type == 2:
        return math.tanh(x)
def acti_func_derivative(x, func_type):
    if func_type == 0:
        return x * (1 - x)
    if func_type == 1:
        return 1 if x > 0 else 0
    if func_type == 2:
        return 1 - x ** 2

class NeuralNetwork():
    def __init__(self, input_size, para_sizes, func_types):
        self.inputs = [0] * input_size
        prev_size = input_size
        cur_size = 0
        self.weights = []
        self.biases = []
        self.func_types = func_types
        self.outputs = []
        self.delta_weights = []
        self.delta_biases = []
        self.delta_outputs = []


        for cur_size in para_sizes:
            self.weights.append([[random.uniform(-0.5, 0.5) for i in range(prev_size)] for j in range(cur_size)])
            self.biases.append([random.uniform(-0.5, 0.5) for j in range(cur_size)])
            self.outputs.append([0 for j in range(cur_size)])
            self.delta_weights.append([[0 for i in range(prev_size)] for j in range(cur_size)])
            self.delta_biases.append([0 for j in range(cur_size)])
            self.delta_outputs.append([0 for j in range(cur_size)])

    def forward(self, inputs):
        cur_inputs = inputs
        self.outputs[0] = [acti_func(sum([cur_inputs[i] * self.weights[0][j][i] for i in range(len(cur_inputs))]) + self.biases[0][j], self.func_types[0]) for j in range(len(self.weights[0]))]
        for layer in range(1, len(self.weights)):
            cur_inputs = self.outputs[layer - 1]
            self.outputs[layer] = [acti_func(sum([cur_inputs[i] * self.weights[layer][j][i] for i in range(len(cur_inputs))]) + self.biases[layer][j], self.func_types[layer]) for j in range(len(self.weights[layer]))]
        return self.outputs[-1]
    def backward(self, target_outputs, learning_rate):
        # Calculate output layer delta
        output_layer = len(self.weights) - 1
        for j in range(len(self.outputs[output_layer])):
            error = target_outputs[j] - self.outputs[output_layer][j]
            self.delta_outputs[output_layer][j] = error * acti_func_derivative(self.outputs[output_layer][j], self.func_types[output_layer])
        # Calculate hidden layer deltas
        for layer in range(output_layer - 1, -1, -1):
            for j in range(len(self.outputs[layer])):
                error = sum([self.delta_outputs[layer + 1][k] * self.weights[layer + 1][k][j] for k in range(len(self.weights[layer + 1]))])
                self.delta_outputs[layer][j] = error * acti_func_derivative(self.outputs[layer][j], self.func_types[layer])
        # Update weights and biases
        for layer in range(len(self.weights)):
            cur_inputs = self.inputs if layer == 0 else self.outputs[layer - 1]
            for j in range(len(self.weights[layer])):
                for i in range(len(cur_inputs)):
                    self.delta_weights[layer][j][i] += learning_rate * self.delta_outputs[layer][j] * cur_inputs[i]
                self.delta_biases[layer][j] += learning_rate * self.delta_outputs[layer][j]
        # Apply weight and bias updates
        for layer in range(len(self.weights)):
            for j in range(len(self.weights[layer])):
                for i in range(len(self.weights[layer][j])):
                    self.weights[layer][j][i] += self.delta_weights[layer][j][i]
                    self.delta_weights[layer][j][i] = 0  # Reset delta after update
                self.biases[layer][j] += self.delta_biases[layer][j]
                self.delta_biases[layer][j] = 0  # Reset delta after update

nn = NeuralNetwork(2, [2, 1], [2, 0])

for epoch in range(10000):
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[1], [1], [0], [0]]  # XOR problem
    for i in range(len(inputs)):
        nn.inputs = inputs[i]
        nn.forward(nn.inputs)
        nn.backward(targets[i], learning_rate=0.5)
for i in range(len(inputs)):
    nn.inputs = inputs[i]
    output = nn.forward(nn.inputs)
    print(f"Input: {inputs[i]}, Predicted Output: {output}, Target: {targets[i]}")