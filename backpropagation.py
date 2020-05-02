# -*- coding: utf-8 -*-

import numpy as np
import neuralnetwork as nn
import mnist

class trainer:
    def __init__(self):
        self.network = nn.network()
        self.__mnistLoaded = False
        self.__miniBatchSize = []
        self.__inputSize = []
        self.__scaleFactor = []
        self.__nIterations = []
    
    def initialiseNetwork(self, structure, *seed):
        self.network.setStructure(structure)
        if len(seed[0])==0:
            np.random.seed()
        else:
            np.random.seed(seed[0])
        for i in range(len(structure)-1):
            self.network.setConnectionWeights(i+1, np.random.random((structure[i+1],structure[i]))*2-1)
            self.network.setNeuronBias(i+1, range(structure[i+1]), np.random.random(structure[i+1])*2-1)
    
    def setNetwork(self, network):
        self.network = network
    
    def getNetwork(self):
        return self.network
    
    def setMiniBatchSize(self, miniBatchSize):
        self.__miniBatchSize = miniBatchSize
    
    def getMiniBatchSize(self):
        return self.__miniBatchSize
    
    def setInputSize(self, inputSize):
        self.__inputSize = inputSize
    
    def getInputSize(self):
        return self.__inputSize
    
    def setIterations(self, nIterations):
        self.__nIterations = nIterations
    
    def getIterations(self):
        return self.__nIterations
    
    def setGradientScaleFactor(self, scaleFactor):
        self.__scaleFactor = scaleFactor
    
    def getGradientScaleFactor(self):
        return self.__scaleFactor
    
    def checkMNISTload(self):
        return self.__mnistLoaded
    
    def loadMNIST(self):
        self.mnistData = mnist.database()
        self.__mnistLoaded = True
    
    def cost(self, inputLayer, desiredOutput):
        self.network.setNeuronActivation(0, range(self.network.getStructure()[0]), inputLayer)
        self.network.evaluate()
        return np.sum((self.network.getNeuronActivation(self.network.getStructure()[-1])-desiredOutput)**2)