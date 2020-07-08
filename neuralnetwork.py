# -*- coding: utf-8 -*-

import numpy as np
from struct import pack, unpack

class network:
    def __init__(self):
        self.__neuronsPerLayer = []
        self.__neuronActivation = []
        self.__neuronBias = []
        self.__neuronConnectionWeights = []
        self.__networkComplete = False
        
    def __sigmoid(self, x):
        # logistic function
        return (1 / (1 + np.exp(-x)))
    
    def evaluate(self):
        try:
            for layerIndex in range(len(self.__neuronsPerLayer)-1):
                z=self.__neuronConnectionWeights[layerIndex+1] @ self.__neuronActivation[layerIndex] + self.__neuronBias[layerIndex+1]
                self.__neuronActivation[layerIndex+1] = self.__sigmoid(z)
            return True
        except:
            return False
    
    def setStructure(self, structure):
        self.__neuronsPerLayer = structure
        self.__neuronActivation = []
        for layerSize in structure:
            self.__neuronActivation.append(np.zeros(layerSize))
        self.__neuronBias = []
        self.__neuronConnectionWeights = []
        for layerSize in structure:
            self.__neuronBias.append(np.zeros(layerSize))
            self.__neuronConnectionWeights.append(np.zeros(1))
        
    def getStructure(self):
        return self.__neuronsPerLayer
    
    def checkNetworkComplete(self):
        self.__networkComplete = True
        if self.__neuronsPerLayer == []:
            self.__networkComplete = False
        if self.__neuronBias == []:
            self.__networkComplete = False
        if self.__neuronConnectionWeights == []:
            self.__networkComplete = False
        for i in range(len(self.__neuronBias)):
            if self.__neuronsPerLayer[i] != np.shape(self.__neuronBias[i])[0]:
                self.__networkComplete = False
        for i in range(len(self.__neuronConnectionWeights)-1):
            if np.shape(self.__neuronConnectionWeights[i+1])[0] != self.__neuronsPerLayer[i+1]:
                self.__networkComplete = False
            if np.shape(self.__neuronConnectionWeights[i+1])[1] != self.__neuronsPerLayer[i]:
                self.__networkComplete = False
        return self.__networkComplete

    def getNetworkComplete(self):
        return self.__networkComplete

    def setNeuronActivation(self, layer, neuron, activation):
        self.__neuronActivation[layer][neuron] = activation
    
    def getNeuronActivation(self, layer, neuron):
        return self.__neuronActivation[layer][neuron]
    
    def setNeuronBias(self, layer, neuron, bias):
        self.__neuronBias[layer][neuron] = bias
    
    def getNeuronBias(self, layer, neuron):
        return self.__neuronBias[layer][neuron]
    
    def setConnectionWeights(self, layer, weights):
        self.__neuronConnectionWeights[layer+1] = weights
        
    def getConnectionWeights(self, layer, *ID):
        if len(ID)==2:
            return self.__neuronConnectionWeights[layer+1][ID[0]][ID[1]]
        else:
            return self.__neuronConnectionWeights[layer+1]

def saveNetwork(networkToSave, fileName):
    if not fileName.endswith('.nn'):
        fileName += '.nn'
    with open(fileName, 'wb') as f:
        f.write(pack('B', len(networkToSave.getStructure())))
        for layerSize in networkToSave.getStructure():
            f.write(pack('<H', layerSize))
        for i in range(len(networkToSave.getStructure())-1):
            for j in range(networkToSave.getStructure()[i+1]):
                for k in range(networkToSave.getStructure()[i]):
                    f.write(pack('<f', networkToSave.getConnectionWeights(i, j, k)))
            for j in range(networkToSave.getStructure()[i+1]):
                f.write(pack('<f', networkToSave.getNeuronBias(i+1, j)))
                
def loadNetwork(fileName):
    try:
        with open(fileName, 'rb') as f:
            nLayers = unpack('B', f.read(1))[0]
            neuronsPerLayer = unpack('{}H'.format(nLayers), f.read(2*nLayers))
            loadedNetwork = network()
            loadedNetwork.setStructure(neuronsPerLayer)
            for i in range(len(neuronsPerLayer)-1):
                weights = np.array(unpack('<{}f'.format(neuronsPerLayer[i]*neuronsPerLayer[i+1]), f.read(4*neuronsPerLayer[i]*neuronsPerLayer[i+1]))).reshape((neuronsPerLayer[i+1],neuronsPerLayer[i]))
                loadedNetwork.setConnectionWeights(i, weights)
                biases = np.array(unpack('<{}f'.format(neuronsPerLayer[i+1]), f.read(4*neuronsPerLayer[i+1])))
                loadedNetwork.setNeuronBias(i+1, range(neuronsPerLayer[i+1]), biases)
        return loadedNetwork
    except:
        return