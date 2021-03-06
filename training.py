# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import simpledialog

import time
import numpy as np
from re import findall
from os import remove

import backpropagation as bp
import neuralnetwork as nn
import mnist
import userInteractivity as UI

from Tooltip import CanvasTooltip

class trainingGUI:
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
        self.batchSizeInput.insert(0, '50')
        self.batchSizeInput.grid(row=4, column=1, sticky='W')
        
        tk.Label(text='Images per epoch:').grid(row=5, column=0)
        self.totalSizeInput = tk.Entry()
        self.totalSizeInput.insert(0, '10000')
        self.totalSizeInput.grid(row=5, column=1, sticky='W')
        
        tk.Label(text='Number of training epochs:').grid(row=6, column=0)
        self.epochInput = tk.Entry()
        self.epochInput.insert(0, '150')
        self.epochInput.grid(row=6, column=1, sticky='W')
        
        tk.Label(text='Learning rate:').grid(row=7, column=0)
        self.learningRateInput = tk.Entry()
        self.learningRateInput.insert(0, '0.1')
        self.learningRateInput.grid(row=7, column=1, sticky='W')
        
        tk.Label(text='Regularisation constant:').grid(row=8,column=0)
        self.regularisationConst = tk.Entry()
        self.regularisationConst.insert(0, '0')
        self.regularisationConst.grid(row=8, column=1, sticky='W')
        
        # TODO: add functionality to these
        tk.Button(text='Configure Grid Search', state='disabled').grid(row=9, column=0)
        tk.Button(text='Run Grid Search', state='disabled').grid(row=9, column=1)
        
        tk.Button(text='Train Network', command=self.__trainNetworkButtonHandler).grid(row=10, column=0)
        tk.Button(text='Train Network, Saving Progress', command=self.__trainSavingProgress).grid(row=10, column=1)
        
        self.trainingProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.trainingProgressBar.grid(row=11, column=0, columnspan=2, sticky='EW')
        
        tk.Button(text='Save Network', command=self.__saveButtonHandler).grid(row=12, column=0, columnspan=2)
        
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
        self.__mnistTrainingLoaded = False
        self.__confusionMatrix = [[[] for i in range(10)] for j in range(10)]
        self.__images = []
        self.__trainingIndices = list(range(60000))
        self.UIelements = UI.elements({'rootWindow': root, \
                                       'progressBarWidget': self.trainingProgressBar, \
                                       'messageLog': self.messageLog, \
                                       'graphCanvas': self.graphingCanvas})
        
        self.__drawConfusionMatrix()
        self.UIelements.drawGraphs([], [], [])
        
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
            self.UIelements.writeToLog('ERROR: No network to train, intialise or load from file.\n')
            return
        
        if self.trainer.getNetwork().getStructure()[0] != 784 or self.trainer.getNetwork().getStructure()[-1] != 10:
            self.UIelements.writeToLog('ERROR: Network must have input size of 784 and output size of 10.\n')
            return
        
        try:
            miniBatchSize = int(self.batchSizeInput.get())
            self.trainer.setMiniBatchSize(miniBatchSize)
        except:
            self.UIelements.writeToLog('ERROR: Mini-batch size must be an integer.\n')
            return
        if miniBatchSize > 60000:
            self.UIelements.writeToLog('ERROR: Mini-batch size must be less than 60,000.\n')
            return
        
        try:
            inputSize = int(self.totalSizeInput.get())
            self.trainer.setInputSize(inputSize)
        except:
            self.UIelements.writeToLog('ERROR: Images per epoch must be an integer.\n')
            return
        if inputSize > 60000:
            self.UIelements.writeToLog('ERROR: Images per epoch must be less than 60,000.\n')
            return
        if inputSize < miniBatchSize:
            self.UIelements.writeToLog('ERROR: Images per epoch must be greater than mini-batch size.\n')
            return
        
        try:
            nEpochs = int(self.epochInput.get())
            self.trainer.setEpochs(nEpochs)
        except:
            self.UIelements.writeToLog('ERROR: Number of training epochs must be an integer.\n')
            return
        if not nEpochs > 0:
            self.UIelements.writeToLog('ERROR: Number of training epochs must be greater than 0.\n')
            return
        
        try:
            learningRate = float(self.learningRateInput.get())
            self.trainer.setLearningRate(learningRate)
        except:
            self.UIelements.writeToLog('ERROR: Learning rate must be a real number.\n')
            return
        if not learningRate > 0:
            self.UIelements.writeToLog('ERROR: Learning rate must be positive.\n')
            return
        
        try:
            regularisationConst = float(self.regularisationConst.get())
            self.trainer.setRegularisationConst(regularisationConst)
        except:
            self.UIelements.writeToLog('ERROR: Regularisation constant must be a real number.\n')
            return
        if not regularisationConst >= 0:
            self.UIelements.writeToLog('ERROR: Regularisation constant must be positive.\n')
            return
        
        self.UIelements.writeToLog('Verifying back-propagation trainer...')
        if self.verboseLog.get():
            self.UIelements.writeToLog('\nNetwork structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] + '\n')
            self.UIelements.writeToLog('Mini-batch size is {:,}\n'.format(miniBatchSize))
            self.UIelements.writeToLog('Images per epoch is {:,}\n'.format(inputSize))
            self.UIelements.writeToLog('Number of training epochs is {:,}\n'.format(nEpochs))
            self.UIelements.writeToLog('Total evaluations of neural network will be {:,} operations\n'.format((inputSize-(inputSize%miniBatchSize))*nEpochs))
            self.UIelements.writeToLog('Learning rate is {}\n'.format(learningRate))
            self.UIelements.writeToLog('Regularisation constant is {}\n'.format(regularisationConst))
            self.UIelements.writeToLog('Verifying back-propagation trainer...')
        self.UIelements.writeToLog('done.\n\n')
        
        return True
    
    def __loadMNISTdatabase(self):
        if not self.__mnistTrainingLoaded:
            self.UIelements.writeToLog('Loading MNIST training database to memory...')
            self.UIelements.updateProgressBar(1)
            mnistDatabase = mnist.database()
            trainingImages, trainingLabels = mnistDatabase.getData(range(60000), 'training')
            self.trainer.setTrainingSet(trainingImages, trainingLabels)
            validationImages, validationLabels = mnistDatabase.getData(range(10000), 'test')
            self.trainer.setValidationSet(validationImages, validationLabels)
            self.__mnistTrainingLoaded = True
            self.UIelements.writeToLog('done.\n\n')
            self.UIelements.updateProgressBar(100)
            time.sleep(0.2)
            self.trainingProgressBar['value'] = 0
        else:
            self.UIelements.writeToLog('MNIST training database already loaded into memory.\n\n')

    def __trainNetworkButtonHandler(self):
        if not self.__checkUserInputForTrainer():
            return
        self.__trainNetwork()
    
    def __trainNetwork(self, *saveIntervalsInfo):
        self.UIelements.drawGraphs([], [], [])
        
        self.__loadMNISTdatabase()
        
        if self.seedCheckVar.get() == 0:
            try:
                np.random.seed(int(self.seedInput.get()))
            except:
                self.UIelements.writeToLog('WARNING: Random seed must be an integer, ignoring entered value.\n')
                np.random.seed()
        else:
            np.random.seed()
        
        startTime = time.time()
        
        if len(saveIntervalsInfo)==0:
            self.trainer.run(rootWindow=root, \
                             progressBarWidget=self.trainingProgressBar, \
                             messageLog=self.messageLog, \
                             graphCanvas=self.graphingCanvas)
        elif len(saveIntervalsInfo)==2:
            self.trainer.run(rootWindow=root, \
                             progressBarWidget=self.trainingProgressBar, \
                             messageLog=self.messageLog, \
                             graphCanvas=self.graphingCanvas, \
                             saveIntervals=saveIntervalsInfo[0], \
                             saveFileRoot=saveIntervalsInfo[1])
        
        trainingDurationTotalSeconds = time.time()-startTime
        trainingDurationString = ''
        trainingDurationDays = int(trainingDurationTotalSeconds / (60*60*24))
        if trainingDurationDays > 0:
            if trainingDurationDays == 1:
                trainingDurationString += '{} day, '.format(trainingDurationDays)
            else:
                trainingDurationString += '{} days, '.format(trainingDurationDays)
        trainingDurationHours = int((trainingDurationTotalSeconds - trainingDurationDays * (60*60*24)) / (60*60))
        if trainingDurationHours > 0 or trainingDurationString != '':
            if trainingDurationHours == 1:
                trainingDurationString += '{} hour, '.format(trainingDurationHours)
            else:
                trainingDurationString += '{} hours, '.format(trainingDurationHours)
        trainingDurationMinutes = int((trainingDurationTotalSeconds - trainingDurationHours * (60*60) - trainingDurationDays * (60*60*24)) / (60))
        if trainingDurationMinutes > 0 or trainingDurationString != '':
            if trainingDurationMinutes == 1:
                trainingDurationString += '{} minute, '.format(trainingDurationMinutes)
            else:
                trainingDurationString += '{} minutes, '.format(trainingDurationMinutes)
        trainingDurationSeconds = trainingDurationTotalSeconds % 60
        trainingDurationString += '{:.1f} seconds.\n\n'.format(trainingDurationSeconds)
        self.UIelements.writeToLog('\nTraining complete.\nDuration ' + trainingDurationString)
        self.UIelements.updateProgressBar(0)
        
        self.__evaluateNetwork()
    
    def __updateEvaluateProgressBar(self, progress):
        self.evaluateProgressBar['value'] = progress
        root.update()
    
    def __evaluateNetwork(self):
        if self.trainer.getNetwork().getStructure() == []:
            self.UIelements.writeToLog('ERROR: No network to evaluate, intialise or load from file.\n')
            return
        
        self.UIelements.writeToLog('Verifying network structure...')
        if self.verboseLog.get():
            self.UIelements.writeToLog('\nNetwork structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] + '\n')

        if self.trainer.getNetwork().getStructure()[0] != 784 or self.trainer.getNetwork().getStructure()[-1] != 10:
            self.UIelements.writeToLog('ERROR: Network must have input size of 784 and output size of 10.\n')
            return
        
        if self.verboseLog.get():
            self.UIelements.writeToLog('Verifying network structure...')
        self.UIelements.writeToLog('done.\n\n')

        if not self.__mnistTestData:
            self.UIelements.writeToLog('Loading MNIST testing database to memory...')
            self.__mnistTestData = mnist.database(True)
            self.UIelements.writeToLog('done.\n\n')
        else:
            self.UIelements.writeToLog('MNIST testing database already loaded into memory.\n\n')
        
        self.__confusionMatrix = [[[] for i in range(10)] for j in range(10)]
        
        self.performanceLabelContent.set('Overall network accuracy: #')
        self.UIelements.writeToLog('Evaluating neural network against MNIST testing database...')
        
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

        self.UIelements.writeToLog('done.\n\n')
        self.__updateEvaluateProgressBar(0)
        
        self.performanceLabelContent.set('Overall network accuracy: {:.2%}'.format(networkAccuracy))
        self.__drawConfusionMatrix()
    
    def __trainSavingProgress(self):
        intervalsString = simpledialog.askstring('Progress Intervals', 'Save network after which training epochs? (csv list)', parent=root)
        if intervalsString is None:
            return
        intervalsList = findall('[0-9]+', intervalsString)
        if len(intervalsList)==0:
            return
        intervals = []
        for i in intervalsList:
            intervals.append(int(i))
        if not self.__checkUserInputForTrainer():
            return
        if max(intervals) > self.trainer.getEpochs():
            messagebox.showerror('Error', 'Save intervals must all be less than number of Epochs.')
            return

        file = filedialog.asksaveasfile(filetypes=(('nn files', '\*.nn'),))
        if file is None:
            return
        file.close()
        remove(file.name)
        
        self.__trainNetwork(intervals, file.name)
    
    def __clearLog(self):
        self.messageLog.configure(state='normal')
        self.messageLog.delete(1.0, tk.END)
        self.messageLog.configure(state='disabled')
    
    def __initialiseNetwork(self):
        hiddenLayersStr = findall('[0-9]+', self.structureInput.get())
        if len(hiddenLayersStr) > 253:
            self.UIelements.writeToLog('ERROR: Network has too many hidden layers (number of hidden layers must be less than 253).\n')
            return
        structure = [28*28]
        for layerStr in hiddenLayersStr:
            if int(layerStr) > 65535:
                self.UIelements.writeToLog('ERROR: Hidden layer size too large (neurons per layer must be less than 65,536).\n')
                return
            structure.append(int(layerStr))
        structure.append(10)
        seed = []
        if self.seedCheckVar.get() == 0:
            try:
                seed.append(int(self.seedInput.get()))
            except:
                self.UIelements.writeToLog('WARNING: Random seed must be an integer, ignoring entered value.\n')
        self.trainer.initialiseNetwork(structure, seed)
        self.UIelements.writeToLog('Initialised random network with structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] +'.\n')
    
    def __loadNetwork(self):
        file = filedialog.askopenfile(title='Select neural network', filetypes=(('neural network files','*.nn'),))
        if file is None:
            return
        
        network = nn.loadNetwork(file.name)
        if network is None:
            messagebox.showerror('Error', 'Error processing file')
            return
        
        structure = network.getStructure()
        if structure[0] != 784 or structure[-1] != 10:
            messagebox.showerror('Error', 'Neural network has wrong number\nof input (784) or output (10) nodes')
            return

        self.trainer.setNetwork(network)
        self.UIelements.writeToLog('Loaded network with structure: ' + str(network.getStructure())[1:-1] +' from file \"' + file.name[file.name.rfind('/')+1:] + '\".\n')

    def __randomiseCheck(self):
        if self.seedCheckVar.get() == 0:
            self.seedInput.configure(state='normal')
        else:
            self.seedInput.configure(state='disabled')
    
    def __saveButtonHandler(self):
        file = filedialog.asksaveasfile(filetypes=(('nn files', '\*.nn'),))
        if file is None:
            return
        file.close()
        remove(file.name)
        nn.saveNetwork(self.trainer.getNetwork(), file.name)
        displayFileName = file.name[file.name.rfind('/')+1:]
        if not displayFileName[-3:] == '.nn':
            displayFileName += '.nn'
        self.UIelements.writeToLog('Saved network to file \"' + displayFileName + '\".\n')
    
root = tk.Tk()
g = trainingGUI(root)
root.mainloop()