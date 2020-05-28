# -*- coding: utf-8 -*-

import numpy as np
import neuralnetwork as nn
import mnist
import userInteractivity as UI

class trainer:
    def __init__(self):
        self.__network = nn.network()
        self.__mnistLoaded = False
        self.__miniBatchSize = []
        self.__inputSize = []
        self.__learningRate = []
        self.__nEpochs = []
        self.__regularisationConst = []
        
    def initialiseNetwork(self, structure, *seed):
        self.__network.setStructure(structure)
        if len(seed[0])==0:
            np.random.seed()
        else:
            np.random.seed(seed[0])
        for i in range(len(structure)-1):
            self.__network.setConnectionWeights(i+1, np.random.random((structure[i+1],structure[i]))*2-1)
            self.__network.setNeuronBias(i+1, range(structure[i+1]), np.random.random(structure[i+1])*2-1)
    
    def run(self, **kwargs):        
        self.__UIelements = UI.elements(kwargs)
        trainingIndices = list(range(60000))
        costHistory = []
        for epoch in range(int(self.getEpochs())):
            print(np.random.randint(0,100))
            np.random.shuffle(trainingIndices)
            lastProgressUpdate = 0
            self.__UIelements.writeToLog('Executing training epoch {:n} of {:n}...'.format(epoch+1,self.getEpochs()))
            for miniBatch in range(int(self.getInputSize()/self.getMiniBatchSize())):
                totalCost = 0
                for trainingExample in range(self.getMiniBatchSize()):
                    currentIndex = miniBatch*self.getMiniBatchSize()+trainingExample
                    percentProgress = currentIndex / (int(self.getInputSize()/self.getMiniBatchSize())*self.getMiniBatchSize())
                    if percentProgress > lastProgressUpdate + 0.005:
                        self.__UIelements.updateProgressBar(percentProgress*100)
                        lastProgressUpdate = percentProgress
                    trainingImage, actualLabel = self.__mnistData.getData(trainingIndices[currentIndex], 'training')
                    exampleCost = self.__exampleCost(trainingImage.reshape(784), actualLabel)
                    totalCost += exampleCost
                    # TODO some actual back propagation
                
                costHistory.append(totalCost / self.getMiniBatchSize())
            self.__UIelements.drawGraphs(costHistory)
            self.__UIelements.writeToLog('done.\n')
    
    def setNetwork(self, network):
        self.__network = network
    
    def getNetwork(self):
        return self.__network
    
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
        self.__mnistData = mnist.database()
        self.__mnistLoaded = True
    
    def __exampleCost(self, inputLayer, desiredOutput):
        self.__network.setNeuronActivation(0, range(self.__network.getStructure()[0]), inputLayer)
        self.__network.evaluate()
        target = np.zeros(self.__network.getStructure()[-1])
        target[desiredOutput] = 1
        return np.sum((self.__network.getNeuronActivation(len(self.__network.getStructure())-1, range(self.__network.getStructure()[-1]))-target)**2)