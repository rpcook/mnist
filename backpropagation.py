# -*- coding: utf-8 -*-

import numpy as np
import nn

class trainer:
    def __init__(self):
        self.network = nn.network()
        pass
    
    def setNetwork(self, network):
        self.network = network
    
    def getNetwork(self):
        return self.network
    
    def cost(self, image, result):
        self.network.setNeuronActivation(0, range(self.network.getStructure()[0]), image)
        self.network.evaluate()
        return np.sum((self.network.getNeuronActivation(self.network.getStructure()[-1])-result)**2)