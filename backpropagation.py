# -*- coding: utf-8 -*-

import numpy as np
import neuralnetwork as nn
from time import time

class trainer:
    def __init__(self):
        self.network = nn.network()
    
    def initialiseNetwork(self, structure, *seed):
        self.network.setStructure(structure)
        if len(seed)==0:
            seed=int((time()-int(time()))*(10**6))
        np.random.seed(seed)
        for i in range(len(structure)-1):
            self.network.setConnectionWeights(i+1, np.random.random((structure[i+1],structure[i]))*2-1)
            self.network.setNeuronBias(i+1, range(structure[i+1]), np.random.random(structure[i+1])*2-1)
    
    def setNetwork(self, network):
        self.network = network
    
    def getNetwork(self):
        return self.network
    
    def cost(self, inputLayer, desiredOutput):
        self.network.setNeuronActivation(0, range(self.network.getStructure()[0]), inputLayer)
        self.network.evaluate()
        return np.sum((self.network.getNeuronActivation(self.network.getStructure()[-1])-desiredOutput)**2)