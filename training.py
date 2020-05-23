# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import messagebox
from struct import pack
import backpropagation as bp
import neuralnetwork as nn
from struct import unpack
import numpy as np
from re import findall
import time
import mnist
from Tooltip import CanvasTooltip

class testingGUI:
    def __init__(self, master):
        self.master = master
        master.title('MNIST Neural Network Training Interface')
        
        tk.Label(text='Hidden layer sizes:').grid(row=0, column=0)
        self.structureInput = tk.Entry()
        self.structureInput.insert(0,'16, 16')
        self.structureInput.grid(row=0, column=1, sticky='W')
        
        tk.Label(text='Random seed:').grid(row=1, column=0)
        self.seedInput = tk.Entry()
        self.seedInput.insert(0, '0')
        self.seedInput.grid(row=1, column=1, sticky='W')
        self.seedCheckVar = tk.IntVar()
        tk.Checkbutton(text='Randomise', variable=self.seedCheckVar, command=self.__randomiseCheck).grid(row=2, column=1, sticky='W')
        
        tk.Button(text='Initialise Network', command=self.__initialiseNetwork).grid(row=3, column=0)
        tk.Button(text='Load Network', command=self.__loadNetwork).grid(row=3, column=1)
        
        tk.Label(text='Mini-batch size:').grid(row=4, column=0)
        self.batchSizeInput = tk.Entry()
        self.batchSizeInput.insert(0, '100')
        self.batchSizeInput.grid(row=4, column=1, sticky='W')
        
        tk.Label(text='Total images to use:').grid(row=5, column=0)
        self.totalSizeInput = tk.Entry()
        self.totalSizeInput.insert(0, '60000')
        self.totalSizeInput.grid(row=5, column=1, sticky='W')
        
        tk.Label(text='Number of training epochs:').grid(row=6, column=0)
        self.epochInput = tk.Entry()
        self.epochInput.insert(0, '1')
        self.epochInput.grid(row=6, column=1, sticky='W')
        
        tk.Label(text='Learning rate:').grid(row=7, column=0)
        self.learningRateInput = tk.Entry()
        self.learningRateInput.insert(0, '0.1')
        self.learningRateInput.grid(row=7, column=1, sticky='W')
        
        tk.Label(text='Regularisation constant:').grid(row=8,column=0)
        self.regularisationConst = tk.Entry()
        self.regularisationConst.insert(0, '0.0')
        self.regularisationConst.grid(row=8, column=1, sticky='W')
        
        # TODO: add functionality to these
        tk.Button(text='Configure Grid Search', state='disabled').grid(row=9, column=0)
        tk.Button(text='Run Grid Search', state='disabled').grid(row=9, column=1)
        
        tk.Button(text='Train Network', command=self.__trainNetwork).grid(row=10, column=0)
        tk.Button(text='Train Network & Evaluate Performance', command=self.__trainAndEvaluateNetwork).grid(row=10, column=1)
        
        self.trainingProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.trainingProgressBar.grid(row=11, column=0, columnspan=2, sticky='EW')
        
        tk.Button(text='Save Network', command=self.__saveButtonHandler).grid(row=12, column=0)
        tk.Button(text='Save Network & Log').grid(row=12, column=1)
        
        self.verboseLog = tk.IntVar()
        self.verboseLog.set(1)
        tk.Checkbutton(text='Verbose logging', variable=self.verboseLog).grid(row=13, column=0, sticky='W')

        self.messageLog = scrolledtext.ScrolledText(height=12, width=70, wrap=tk.WORD, state='disabled', font=('Arial',9))
        self.messageLog.grid(row=14, column=0, columnspan=2)
        self.messageLog.tag_config('error', foreground='red')
        self.messageLog.tag_config('warning', foreground='orange')
        
        self.graphingCanvas = tk.Canvas(width=510, height=150)
        self.graphingCanvas.grid(row=15, column=0, columnspan=2)
        
        ttk.Separator(orient=tk.VERTICAL).grid(row=0, column=2, rowspan=16, sticky='NS')
        
        tk.Button(text='Evaluate Network\nPerformance', command=self.__evaluateNetwork).grid(row=0, column=3, rowspan=2)

        self.evaluateProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.evaluateProgressBar.grid(row=0, column=4)
        
        self.performanceLabelContent = tk.StringVar()
        self.performanceLabelContent.set('Overall network accuracy: #')
        self.performanceLabel = tk.Label(textvariable=self.performanceLabelContent)
        self.performanceLabel.grid(row=1, column=4)
        
        self.__gridSize = 55
        self.confusionCanvas = tk.Canvas(width=11*self.__gridSize, height=11*self.__gridSize)
        self.confusionCanvas.grid(row=2, column=3, rowspan=14, columnspan=2)
        
        self.trainer = bp.trainer()
        self.__mnistTestData = False
        self.__confusionMatrix = [[[] for i in range(10)] for j in range(10)]
        self.__images = []
        self.__trainingIndices = list(range(60000))
        self.__costHistory = []
        
        self.__drawConfusionMatrix()
        self.__drawCostGraph()
        
    def __drawCostGraph(self):
        gc = self.graphingCanvas
        gc.delete('all')
        gc.create_text(15,75,angle=90,text='Loss')
        boldLine = '#aaaaaa'
        softLine = '#dddddd'
        costLine = '#0000ff'
        if len(self.__costHistory) < 2:
            numMajorGrids = 1
        else:
            numMajorGrids = int(np.ceil(np.log10(10/min(self.__costHistory))))
        
        gc.create_line(30,8,30,145)
        for i in range(numMajorGrids+1):
            gc.create_line(31, 8+i*(136/numMajorGrids), 500, 8+i*(136/numMajorGrids), fill=boldLine)
        for i in range(numMajorGrids):
            for j in range(2,10):
                yPos = 8+(i+1-np.log10(j))*(136/numMajorGrids)
                gc.create_line(31, yPos, 500, yPos, fill=softLine)
                
        if len(self.__costHistory) > 1:
            for i in range(len(self.__costHistory)-1):
                yPosStart = 8 + 136*((1 - np.log10(self.__costHistory[i])) / numMajorGrids)
                yPosStop  = 8 + 136*((1 - np.log10(self.__costHistory[i+1])) / numMajorGrids)
                gc.create_line(31+i*(470/(len(self.__costHistory)-1)), yPosStart, 31+(i+1)*(470/(len(self.__costHistory)-1)), yPosStop, fill=costLine)
        root.update()
    
    def __drawConfusionMatrix(self, drawNumbers=True):
        cm = self.__confusionMatrix
        cc = self.confusionCanvas
        gd = self.__gridSize
        cc.delete('all')
        self.__images = []
        cc.create_text(gd*6,gd*0.3,text='Actual digit')
        cc.create_text(gd*0.3,gd*6,angle=90,text='Predicted digit')
        for i in range(10):
            cc.create_text((i+1.5)*gd,gd*0.7, text=str(i))
            cc.create_text(gd*0.7,(i+1.5)*gd, text=str(i), angle=90)
        
        entriesInConfusionMatrix = 0
        for i in range(10):
            for j in range(10):
                entriesInConfusionMatrix += len(cm[i][j])
        
        if entriesInConfusionMatrix > 0:
            cellColour = [['#' for x in range(10)] for x in range(10)] 
            for i in range(10):
                totalActual = 0
                for j in range(10):
                    totalActual += len(cm[i][j])
                for j in range(10):
                    if totalActual == 0:
                        cellHighlight = 0
                    elif i == j:
                        cellHighlight = 10 * len(cm[i][j]) / totalActual
                    else:
                        cellHighlight = 100 * len(cm[i][j]) / totalActual
                    brightness = 255-min(int(10*(cellHighlight**1.4)),255)
                    hexValue = '%0.2X' % abs(brightness)
                    if i == j:
                        cellColour[i][j] = '#' + hexValue + 'ff' + hexValue
                    else:
                        cellColour[i][j] = '#' + 'ff' + hexValue + hexValue
        else:
            cellColour = [['#ffffff' for x in range(10)] for x in range(10)] 
        
        for i in range(10):
            for j in range(10):
                self.confusionCanvas.create_rectangle((i+1)*gd,(j+1)*gd,(i+2)*gd,(j+2)*gd, fill=cellColour[i][j])
                if (entriesInConfusionMatrix > 0) and drawNumbers:
                    id_ = cc.create_text((i+1.5)*gd,(j+1.5)*gd, text=format(len(cm[i][j]), 'n'))
                    if len(cm[i][j]) > 0:
                        CanvasTooltip(cc, id_, text=str(cm[i][j]).strip('[]'), wraplength=250+300*(len(cm[i][j])/1000))
        root.update()
    
    def __checkUserInputForTrainer(self):
        self.__clearLog()
        if self.trainer.getNetwork().getStructure() == []:
            self.__writeToLog('ERROR: No network to train, intialise or load from file.\n')
            return
        
        if self.trainer.getNetwork().getStructure()[0] != 784 or self.trainer.getNetwork().getStructure()[-1] != 10:
            self.__writeToLog('ERROR: Network must have input size of 784 and output size of 10.\n')
            return
        
        try:
            miniBatchSize = int(self.batchSizeInput.get())
            self.trainer.setMiniBatchSize(miniBatchSize)
        except:
            self.__writeToLog('ERROR: Mini-batch size must be an integer.\n')
            return
        if miniBatchSize > 60000:
            self.__writeToLog('ERROR: Mini-batch size must be less than 60,000.\n')
            return
        
        try:
            inputSize = int(self.totalSizeInput.get())
            self.trainer.setInputSize(inputSize)
        except:
            self.__writeToLog('ERROR: Total images to use must be an integer.\n')
            return
        if inputSize > 60000:
            self.__writeToLog('ERROR: Total images to use must be less than 60,000.\n')
            return
        if inputSize < miniBatchSize:
            self.__writeToLog('ERROR: Total images to use must be greater than mini-batch size.\n')
            return
        
        try:
            nEpochs = int(self.epochInput.get())
            self.trainer.setEpochs(nEpochs)
        except:
            self.__writeToLog('ERROR: Number of training epochs must be an integer.\n')
            return
        if not nEpochs > 0:
            self.__writeToLog('ERROR: Number of training epochs must be greater than 0.\n')
            return
        
        try:
            learningRate = float(self.learningRateInput.get())
            self.trainer.setLearningRate(learningRate)
        except:
            self.__writeToLog('ERROR: Learning rate must be a real number.\n')
            return
        if not learningRate > 0:
            self.__writeToLog('ERROR: Learning rate must be positive.\n')
            return
        
        try:
            regularisationConst = float(self.regularisationConst.get())
            self.trainer.setRegularisationConst(regularisationConst)
        except:
            self.__writeToLog('ERROR: Regularisation constant must be a real number.\n')
            return
        if not regularisationConst >= 0:
            self.__writeToLog('ERROR: Regularisation constant must be positive.\n')
            return
        
        self.__writeToLog('Verifying back-propagation trainer...')
        if self.verboseLog.get():
            self.__writeToLog('\nNetwork structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] + '\n')
            self.__writeToLog('Mini-batch size is {:,}\n'.format(miniBatchSize))
            self.__writeToLog('Total images to use is {:,}\n'.format(inputSize))
            self.__writeToLog('Number of training epochs is {:,}\n'.format(nEpochs))
            self.__writeToLog('Total evaluations of neural network will be {:,} operations\n'.format((inputSize-(inputSize%miniBatchSize))*nEpochs))
            self.__writeToLog('Learning rate is {}\n'.format(learningRate))
            self.__writeToLog('Regularisation constant is {}\n'.format(regularisationConst))
            self.__writeToLog('Verifying back-propagation trainer...')
        self.__writeToLog('done.\n\n')
        
        return True
    
    def __loadMNISTdatabase(self):
        if not self.trainer.checkMNISTload():
            self.__writeToLog('Loading MNIST training database to memory...')
            self.__updateTrainingProgressBar(1)
            self.trainer.loadMNIST()
            self.__writeToLog('done.\n\n')
            self.__updateTrainingProgressBar(100)
            time.sleep(0.2)
            self.trainingProgressBar['value'] = 0
        else:
            self.__writeToLog('MNIST training database already loaded into memory.\n\n')
    
    def __trainNetwork(self):
        self.__costHistory = []
        self.__drawCostGraph()
        
        if not self.__checkUserInputForTrainer():
            return
        self.__loadMNISTdatabase()
        
        if self.seedCheckVar.get() == 0:
            try:
                np.random.seed(int(self.seedInput.get()))
            except:
                self.__writeToLog('WARNING: Random seed must be an integer, ignoring entered value.\n')
                np.random.seed()
        else:
            np.random.seed()
        
        startTime = time.time()
        
        for iteration in range(int(self.trainer.getEpochs())):
            np.random.shuffle(self.__trainingIndices)
            lastProgressUpdate = 0
            self.__writeToLog('Executing training epoch {:n} of {:n}...'.format(iteration+1,self.trainer.getEpochs()))
            for miniBatch in range(int(self.trainer.getInputSize()/self.trainer.getMiniBatchSize())):
                totalCost = 0
                for trainingExample in range(self.trainer.getMiniBatchSize()):
                    currentIndex = miniBatch*self.trainer.getMiniBatchSize()+trainingExample
                    percentProgress = currentIndex / (int(self.trainer.getInputSize()/self.trainer.getMiniBatchSize())*self.trainer.getMiniBatchSize())
                    if percentProgress > lastProgressUpdate + 0.005:
                        self.__updateTrainingProgressBar(percentProgress*100)
                        lastProgressUpdate = percentProgress
                    trainingImage, actualLabel = self.trainer.mnistData.getData(self.__trainingIndices[currentIndex], 'training')
                    exampleCost = self.trainer.cost(trainingImage.reshape(784), actualLabel)
                    totalCost += exampleCost
                    # TODO some actual back propagation
                
                self.__costHistory.append(totalCost / self.trainer.getMiniBatchSize())
            self.__drawCostGraph()
            self.__writeToLog('done.\n')
        self.__updateTrainingProgressBar(0)
        self.__writeToLog('\nTraining complete. Duration {:.1f}s.\n\n'.format(time.time()-startTime))
    
    def __updateTrainingProgressBar(self, progress):
        self.trainingProgressBar['value'] = progress
        root.update()
    
    def __updateEvaluateProgressBar(self, progress):
        self.evaluateProgressBar['value'] = progress
        root.update()
    
    def __evaluateNetwork(self):
        if self.trainer.getNetwork().getStructure() == []:
            self.__writeToLog('ERROR: No network to evaluate, intialise or load from file.\n')
            return
        
        self.__writeToLog('Verifying network structure...')
        if self.verboseLog.get():
            self.__writeToLog('\nNetwork structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] + '\n')

        if self.trainer.getNetwork().getStructure()[0] != 784 or self.trainer.getNetwork().getStructure()[-1] != 10:
            self.__writeToLog('ERROR: Network must have input size of 784 and output size of 10.\n')
            return
        
        if self.verboseLog.get():
            self.__writeToLog('Verifying network structure...')
        self.__writeToLog('done.\n\n')

        if not self.__mnistTestData:
            self.__writeToLog('Loading MNIST testing database to memory...')
            self.__mnistTestData = mnist.database(True)
            self.__writeToLog('done.\n\n')
        else:
            self.__writeToLog('MNIST testing database already loaded into memory.\n\n')
        
        # self.__confusionMatrix = np.zeros((10,10))
        self.__confusionMatrix = [[[] for i in range(10)] for j in range(10)]
        
        self.__writeToLog('Evaluating neural network against MNIST testing database...')
        
        network = self.trainer.getNetwork()
        for i in range(10000):
            if i%100==0:
                self.__updateEvaluateProgressBar(i/100)
            if i%300==0:
                self.__drawConfusionMatrix(False)
            testImage, actualLabel = self.__mnistTestData.getData(i, 'test')
            network.setNeuronActivation(0, range(784), testImage.reshape(784))
            network.evaluate()
            networkPrediction = np.argmax(network.getNeuronActivation(-1, range(10)))
            self.__confusionMatrix[actualLabel][networkPrediction].append(i)

        totalCorrectPredictions = 0
        for i in range(10):
            totalCorrectPredictions += len(self.__confusionMatrix[i][i])
        networkAccuracy = totalCorrectPredictions / 10000

        self.__writeToLog('done.\n\n')
        self.__updateEvaluateProgressBar(0)
        
        self.performanceLabelContent.set('Overall network accuracy: {:.2%}'.format(networkAccuracy))
        self.__drawConfusionMatrix()
    
    def __trainAndEvaluateNetwork(self):
        self.__trainNetwork()
        self.__evaluateNetwork()
    
    def __writeToLog(self, message):
        self.messageLog.configure(state='normal')
        if len(message)>7:
            if message[0:5]=='ERROR':
                self.messageLog.insert(tk.END, message[0:5], 'error')
                self.messageLog.insert(tk.END, message[5:])
            elif message[0:7]=='WARNING':
                self.messageLog.insert(tk.END, message[0:7], 'warning')
                self.messageLog.insert(tk.END, message[7:])
            else:
                self.messageLog.insert(tk.END, message)
        else:
            self.messageLog.insert(tk.END, message)
        self.messageLog.configure(state='disabled')
        self.messageLog.yview(tk.END)
        root.update()
        
    def __clearLog(self):
        self.messageLog.configure(state='normal')
        self.messageLog.delete(1.0, tk.END)
        self.messageLog.configure(state='disabled')
    
    def __initialiseNetwork(self):
        hiddenLayersStr = findall('[0-9]+', self.structureInput.get())
        structure = [28*28]
        for layerStr in hiddenLayersStr:
            structure.append(int(layerStr))
        structure.append(10)
        seed = []
        if self.seedCheckVar.get() == 0:
            try:
                seed.append(int(self.seedInput.get()))
            except:
                self.__writeToLog('WARNING: Random seed must be an integer, ignoring entered value.\n')
        self.trainer.initialiseNetwork(structure, seed)
        self.__writeToLog('Initialised random network with structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] +'.\n')
    
    def __loadNetwork(self):
        file = filedialog.askopenfile(title='Select neural network', filetypes=(('neural network files','*.nn'),))
        if file is None:
            return
        with open(file.name, 'rb') as f:
            nLayers = unpack('B', f.read(1))[0]
            neuronsPerLayer = unpack('{}H'.format(nLayers), f.read(2*nLayers))
            if neuronsPerLayer[0] != 784 or neuronsPerLayer[3] != 10:
                messagebox.showerror('Error', 'Neural network has wrong number\nof input (784) or output (10) nodes')
                return
            network = nn.network()
            network.setStructure(neuronsPerLayer)
            for i in range(len(neuronsPerLayer)-1):
                weights = np.array(unpack('<{}f'.format(neuronsPerLayer[i]*neuronsPerLayer[i+1]), f.read(4*neuronsPerLayer[i]*neuronsPerLayer[i+1]))).reshape((neuronsPerLayer[i+1],neuronsPerLayer[i]))
                network.setConnectionWeights(i+1, weights)
                biases = np.array(unpack('<{}f'.format(neuronsPerLayer[i+1]), f.read(4*neuronsPerLayer[i+1])))
                network.setNeuronBias(i+1, range(neuronsPerLayer[i+1]), biases)
        self.trainer.setNetwork(network)
        self.__writeToLog('Loaded network with structure: ' + str(network.getStructure())[1:-1] +'.\n')

    def __randomiseCheck(self):
        if self.seedCheckVar.get() == 0:
            self.seedInput.configure(state='normal')
        else:
            self.seedInput.configure(state='disabled')
    
    def __saveButtonHandler(self):
        file = filedialog.asksaveasfile(filetypes=(('nn files', '\*.nn'),))
        if file is None:
            return
        self.__saveNetwork(self.trainer.getNetwork(), file.name)
    
    def __saveNetwork(self, network, fileName):
        with open(fileName, 'wb') as f:
            f.write(pack('B', len(network.getStructure())))
            for layerSize in network.getStructure():
                f.write(pack('<H', layerSize))
            for i in range(len(network.getStructure())-1):
                for j in range(network.getStructure()[i+1]):
                    for k in range(network.getStructure()[i]):
                        f.write(pack('<f', network.getConnectionWeights(i, j, k)))
                for j in range(network.getStructure()[i+1]):
                    f.write(pack('<f', network.getNeuronBias(i, j)))

root = tk.Tk()
g = testingGUI(root)
root.mainloop()