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
        validationIndices = list(range(10000))
        lossHistoryTrainer = []
        lossHistoryValidation = []
        errorHistory = []
        for epoch in range(int(self.getEpochs())):
            self.__UIelements.writeToLog('Executing training epoch {:n} of {:n}...'.format(epoch+1,self.getEpochs()))
            
            deltaSum = [np.array([0])]
            for layerSize in self.__network.getStructure()[1:]:
                deltaSum.append(np.zeros(layerSize))
            grads = [np.array([0])]
            for layer in range(len(self.__network.getStructure())-1):
                grads.append(np.zeros((layer+1, layer)))
                
            # Back propagation / training
            np.random.shuffle(trainingIndices)
            lastProgressUpdate = 0
            totalCost = 0
            for miniBatch in range(int(self.getInputSize()/self.getMiniBatchSize())):
                for trainingExample in range(self.getMiniBatchSize()):
                    currentIndex = miniBatch*self.getMiniBatchSize()+trainingExample
                    percentProgress = currentIndex / (int(self.getInputSize()/self.getMiniBatchSize())*self.getMiniBatchSize())
                    if percentProgress > lastProgressUpdate + 0.005:
                        self.__UIelements.updateProgressBar(percentProgress*100)
                        lastProgressUpdate = percentProgress
                    
                    # forward calculation
                    trainingImage, actualLabel = self.__mnistData.getData(trainingIndices[currentIndex], 'training')
                    exampleCost = self.__exampleCost(trainingImage.reshape(784), actualLabel)
                    totalCost += exampleCost

                    # back propagation of errors
                    deltas = [np.array([0])]
                    for layerSize in self.__network.getStructure()[1:]:
                        deltas.append(np.zeros(layerSize))
                    targetOutputActivations = np.zeros(self.__network.getStructure()[-1])
                    targetOutputActivations[actualLabel] = 1
                    outputActivations = self.__network.getNeuronActivation(len(self.__network.getStructure())-1, range(self.__network.getStructure()[-1]))
                    deltas[len(self.__network.getStructure())-1] = \
                        (outputActivations - targetOutputActivations) * (outputActivations) * (1 - outputActivations)
                    for layer in range(len(self.__network.getStructure())-1,0,-1):
                        layerActivation = self.__network.getNeuronActivation(layer-1, range(self.__network.getStructure()[layer-1]))
                        deltas[layer-1] = (np.transpose(self.__network.getConnectionWeights(layer-1)) @ deltas[layer]) * (layerActivation) * (1 - layerActivation)
                    
                    # accumulators
                    for layer in range(len(self.__network.getStructure())-1):
                        deltaSum[layer+1] += deltas[layer+1]
                        grads[layer+1] += grads[layer+1] # TODO: exterior product of activations and deltas per layer
                    
                    # gradient descent
            lossHistoryTrainer.append(totalCost / self.getInputSize())
            
            # Validation (random subset of test set)
            np.random.shuffle(validationIndices)
            totalCost = 0
            totalErrors = 0
            for validationIndex in range(max(int(self.getInputSize()/60), 500)):
                validationImage, actualLabel = self.__mnistData.getData(validationIndices[validationIndex], 'test')
                testImageCost = self.__exampleCost(validationImage.reshape(784), actualLabel)
                totalCost += testImageCost
                if actualLabel != np.argmax(self.__network.getNeuronActivation(len(self.__network.getStructure())-1, range(self.__network.getStructure()[-1]))):
                    totalErrors += 1
            lossHistoryValidation.append(totalCost / max(int(self.getInputSize()/100), 500))
            errorHistory.append(10 * totalErrors / max(int(self.getInputSize()/100), 500))
            
            self.__UIelements.drawGraphs(lossHistoryTrainer, lossHistoryValidation, errorHistory)
            self.__UIelements.writeToLog('done.\n')
    
    def __exampleCost(self, inputLayer, desiredOutput):
        self.__network.setNeuronActivation(0, range(self.__network.getStructure()[0]), inputLayer)
        self.__network.evaluate()
        target = np.zeros(self.__network.getStructure()[-1])
        target[desiredOutput] = 1
        return np.sum((self.__network.getNeuronActivation(len(self.__network.getStructure())-1, range(self.__network.getStructure()[-1]))-target)**2)
    
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
