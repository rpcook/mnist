# -*- coding: utf-8 -*-

import numpy as np
import neuralnetwork as nn
import userInteractivity as UI

class trainer:
    def __init__(self):
        self.__network = nn.network()
        self.__trainingLoaded = False
        self.__validationLoaded = False
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
            self.__network.setConnectionWeights(i, np.random.random((structure[i+1],structure[i]))*2-1)
            self.__network.setNeuronBias(i+1, range(structure[i+1]), np.random.random(structure[i+1])*2-1)
    
    def run(self, **kwargs):
        if not self.__trainingLoaded:
            raise NameError('Training data not loaded!')
        if not self.__validationLoaded:
            raise NameError('Validation data not loaded!')
        self.__UIelements = UI.elements(kwargs)
        trainingIndices = list(range(len(self.__trainingSetLabels)))
        validationIndices = list(range(len(self.__validationSetLabels)))
        lossHistoryTrainer = []
        lossHistoryValidation = []
        errorHistory = []
        for epoch in range(int(self.getEpochs())):
            self.__UIelements.writeToLog('Executing training epoch {:n} of {:n}...'.format(epoch+1,self.getEpochs()))
            
            # Back propagation / training
            np.random.shuffle(trainingIndices)
            lastProgressUpdate = 0
            totalTrainingCost = 0
            for miniBatch in range(int(self.__inputSize / self.__miniBatchSize)):
                deltaSum = [np.array([0])]
                for layerSize in self.__network.getStructure()[1:]:
                    deltaSum.append(np.zeros(layerSize))
                grads = []
                for layer in range(len(self.__network.getStructure())-1):
                    grads.append(np.zeros((self.__network.getStructure()[layer+1], self.__network.getStructure()[layer])))

                for trainingExample in range(self.__miniBatchSize):
                    currentIndex = miniBatch * self.__miniBatchSize + trainingExample
                    percentProgress = currentIndex / (int(self.__inputSize / self.__miniBatchSize) * self.__miniBatchSize)
                    if percentProgress > lastProgressUpdate + 0.005:
                        self.__UIelements.updateProgressBar(percentProgress*100)
                        lastProgressUpdate = percentProgress
                    
                    # forward calculation
                    trainingImage, actualLabel = self.__getTrainingExample(trainingIndices[currentIndex])
                    exampleCost = self.__exampleCost(trainingImage.reshape(784), actualLabel)
                    totalTrainingCost += exampleCost

                    # back propagation of errors
                    deltas = [np.array([0])]
                    for layerSize in self.__network.getStructure()[1:]:
                        deltas.append(np.zeros(layerSize))
                    targetOutputActivations = np.zeros(self.__network.getStructure()[-1])
                    targetOutputActivations[actualLabel] = 1
                    outputActivations = self.__network.getNeuronActivation(len(self.__network.getStructure())-1, range(self.__network.getStructure()[-1]))
                    deltas[len(self.__network.getStructure())-1] = (outputActivations - targetOutputActivations) * (outputActivations) * (1 - outputActivations)
                    for layer in range(len(self.__network.getStructure())-1,0,-1):
                        layerActivation = self.__network.getNeuronActivation(layer-1, range(self.__network.getStructure()[layer-1]))
                        deltas[layer-1] = (np.transpose(self.__network.getConnectionWeights(layer-1)) @ deltas[layer]) * (layerActivation) * (1 - layerActivation)
                    
                    # accumulators
                    for layer in range(len(self.__network.getStructure())-1):
                        deltaSum[layer+1] += deltas[layer+1]
                        grads[layer] += deltas[layer+1].reshape(len(deltas[layer+1]),1) * self.__network.getNeuronActivation(layer, range(self.__network.getStructure()[layer]))
                    
                # gradient descent
                for layer in range(len(self.__network.getStructure())-1):
                    weights = self.__network.getConnectionWeights(layer)
                    weights -= (self.__learningRate / self.__miniBatchSize) * grads[layer] + self.__regularisationConst * weights
                    self.__network.setConnectionWeights(layer, weights)
                    
                    biases = self.__network.getNeuronBias(layer+1, range(self.__network.getStructure()[layer+1]))
                    biases -= (self.__learningRate / self.__miniBatchSize) * deltaSum[layer+1]
                    self.__network.setNeuronBias(layer+1, range(self.__network.getStructure()[layer+1]), biases)

            lossHistoryTrainer.append(totalTrainingCost / self.__inputSize)
            
            # Validation (random subset of test set)
            np.random.shuffle(validationIndices)
            totalValidationCost = 0
            totalErrors = 0
            validationSetSize = max(int(self.__inputSize/60), 500)
            for validationIndex in range(validationSetSize):
                validationImage, actualLabel = self.__getValidationExample(validationIndices[validationIndex])
                testImageCost = self.__exampleCost(validationImage.reshape(784), actualLabel)
                totalValidationCost += testImageCost
                if actualLabel != np.argmax(self.__network.getNeuronActivation(len(self.__network.getStructure())-1, range(self.__network.getStructure()[-1]))):
                    totalErrors += 1
            lossHistoryValidation.append(totalValidationCost / validationSetSize)
            errorHistory.append(10 * totalErrors / validationSetSize)
            
            self.__UIelements.drawGraphs(lossHistoryTrainer, lossHistoryValidation, errorHistory)
            self.__UIelements.writeToLog('done.\n')
            
            if 'saveIntervals' in kwargs and 'saveFileRoot' in kwargs:
                if kwargs['saveFileRoot'].endswith('.nn'):
                    fileRoot = kwargs['saveFileRoot'][:-len('.nn')]
                else:
                    fileRoot = kwargs['saveFileRoot']
                indexLength = int(np.ceil(np.log10(max(kwargs['saveIntervals'])+1)))
                if epoch+1 in kwargs['saveIntervals']:
                    saveFile = fileRoot + '{:0{}d}.nn'.format(epoch+1, indexLength)
                    nn.saveNetwork(self.__network, saveFile)
    
    def __exampleCost(self, inputLayer, desiredOutput):
        self.__network.setNeuronActivation(0, range(self.__network.getStructure()[0]), inputLayer)
        self.__network.evaluate()
        target = np.zeros(self.__network.getStructure()[-1])
        target[desiredOutput] = 1
        return np.sum((self.__network.getNeuronActivation(len(self.__network.getStructure())-1, range(self.__network.getStructure()[-1]))-target)**2)
    
    def setTrainingSet(self, trainingSetImages, trainingSetLabels):
        self.__trainingSetImages = trainingSetImages
        self.__trainingSetLabels = trainingSetLabels
        self.__trainingLoaded = True
        
    def __getTrainingExample(self, indices):
        images = self.__trainingSetImages[indices]
        labels = self.__trainingSetLabels[indices]
        return images, labels
    
    def setValidationSet(self, validationSetImages, validationSetLabels):
        self.__validationSetImages = validationSetImages
        self.__validationSetLabels = validationSetLabels
        self.__validationLoaded = True
        
    def __getValidationExample(self, indices):
        images = self.__trainingSetImages[indices]
        labels = self.__trainingSetLabels[indices]
        return images, labels
    
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