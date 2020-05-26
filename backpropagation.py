# -*- coding: utf-8 -*-

import numpy as np
import neuralnetwork as nn
import mnist
import time
import userInteractivity as UI

class trainer:
    def __init__(self):
        self.network = nn.network()
        self.__mnistLoaded = False
        self.__miniBatchSize = []
        self.__inputSize = []
        self.__learningRate = []
        self.__nEpochs = []
        self.__regularisationConst = []
        
    def initialiseNetwork(self, structure, *seed):
        self.network.setStructure(structure)
        if len(seed[0])==0:
            np.random.seed()
        else:
            np.random.seed(seed[0])
        for i in range(len(structure)-1):
            self.network.setConnectionWeights(i+1, np.random.random((structure[i+1],structure[i]))*2-1)
            self.network.setNeuronBias(i+1, range(structure[i+1]), np.random.random(structure[i+1])*2-1)
    
    def run(self, **kwargs):        
        self.UIelements = UI.elements(kwargs)
        
        for i in range(10):
            time.sleep(0.3)
            self.UIelements.writeToLog('huzzah iteration {}\n'.format(i))
            self.UIelements.updateProgressBar(10*i)
    
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
    
    def setEpochs(self, nEpochs):
        self.__nEpochs = nEpochs
    
    def getEpochs(self):
        return self.__nEpochs
    
    def setLearningRate(self, learningRate):
        self.__learningRate = learningRate
    
    def getLearningRate(self):
        return self.__learningRate
    
    def setRegularisationConst(self, regularisationConst):
        self.__regularisationConst = regularisationConst
    
    def getRegularisationConst(self):
        return self.__regularisationConst
    
    def checkMNISTload(self):
        return self.__mnistLoaded
    
    def loadMNIST(self):
        self.mnistData = mnist.database()
        self.__mnistLoaded = True
    
    def cost(self, inputLayer, desiredOutput):
        self.network.setNeuronActivation(0, range(self.network.getStructure()[0]), inputLayer)
        self.network.evaluate()
        target = np.zeros(self.network.getStructure()[-1])
        target[desiredOutput] = 1
        return np.sum((self.network.getNeuronActivation(len(self.network.getStructure())-1, range(self.network.getStructure()[-1]))-target)**2)