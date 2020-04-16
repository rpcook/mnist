# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:04:05 2020

@author: Rob.Cook
"""

import numpy as np

def sigmoid(x):
    # logistic function
    return (1 / (1 + np.exp(-x)))

def dsigmoid(x):
    # first derivative of logistic function
    return sigmoid(x)*sigmoid(-x)

class network:
    def __init__(self):
        self.__neuronsPerLayer = []
        self.__neuronActivations = []
    
    def setStructure(self, structure):
        self.__neuronsPerLayer = structure
        self.__neuronActivation = []
        for layerSize in structure:
            self.__neuronActivations.append(np.zeros(layerSize))
        
    def getStructure(self):
        return self.__neuronsPerLayer
    
    def setNeuronActivation(self, layer, neuron, activation):
        self.__neuronActivations[layer][neuron] = activation
    
    def getNeuronActivation(self, layer, neuron):
        return self.__neuronActivations[layer][neuron]