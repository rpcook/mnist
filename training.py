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
import re
import time

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
        
        tk.Label(text='Gradient descent scale-factor:').grid(row=6, column=0)
        self.scaleFactorInput = tk.Entry()
        self.scaleFactorInput.insert(0, '1.0')
        self.scaleFactorInput.grid(row=6, column=1, sticky='W')
        
        tk.Button(text='Train Network', command=self.__trainNetwork).grid(row=7, column=0)
        tk.Button(text='Train Network & Evaluate Performance', command=self.__trainAndEvaluateNetwork).grid(row=7, column=1)
        
        self.trainingProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.trainingProgressBar.grid(row=8, column=0, columnspan=2, sticky='EW')
        
        tk.Button(text='Save Network', command=self.__saveButtonHandler).grid(row=9, column=0)
        tk.Button(text='Save Network & Log').grid(row=9, column=1)
        
        self.messageLog = scrolledtext.ScrolledText(height=10, width=70, wrap=tk.WORD, state='disabled')
        self.messageLog.grid(row=10, column=0, columnspan=2)
        
        ttk.Separator(orient=tk.VERTICAL).grid(row=0, column=2, rowspan=11, sticky='NS')
        
        tk.Button(text='Evaluate Network\nPerformance', command=self.__evaluateNetwork).grid(row=0, column=3, rowspan=2)

        self.evaluateProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.evaluateProgressBar.grid(row=0, column=4)
        
        self.performanceLabelContent = tk.StringVar()
        self.performanceLabelContent.set('Overall network accuracy: #')
        self.performanceLabel = tk.Label(textvariable=self.performanceLabelContent)
        self.performanceLabel.grid(row=1, column=4)
        
        self.__gridSize = 36
        self.confusionCanvas = tk.Canvas(width=11*self.__gridSize, height=11*self.__gridSize)
        self.confusionCanvas.grid(row=2, column=3, rowspan=9, columnspan=2)
        
        self.trainer = bp.trainer()
        self.__confusionMatrix = np.zeros((10,10))
        
        self.__drawConfusionMatrix()
    
    def __drawConfusionMatrix(self):
        cm = self.__confusionMatrix
        cc = self.confusionCanvas
        gd = self.__gridSize
        cc.delete('all')
        cc.create_text(gd*6,gd*0.3,text='Actual digit')
        cc.create_text(gd*0.3,gd*6,angle=90,text='Predicted digit')
        for i in range(10):
            cc.create_text((i+1.5)*gd,gd*0.7, text=str(i))
            cc.create_text(gd*0.7,(i+1.5)*gd, text=str(i), angle=90)
        if np.any(cm):
            # TODO: have some nice R/G colouring here for cells
            cellColour = [['']*10]*10
        else:
            cellColour = [['']*10]*10
        for i in range(10):
            for j in range(10):
                cc.create_rectangle((i+1)*gd,(j+1)*gd,(i+2)*gd,(j+2)*gd, fill=cellColour[i][j])
                if np.any(cm):
                    cc.create_text((i+1.5)*gd,(j+1.5)*gd, text=str(cm[i][j]))
    
    def __trainNetwork(self):
        self.__clearLog()
        if self.trainer.getNetwork().getStructure() == []:
            self.__writeToLog('ERROR: No network to train, intialise or load from file.\n')
            return
        
        try:
            miniBatchSize = int(self.batchSizeInput.get())
            self.trainer.setMiniBatchSize(miniBatchSize)
        except:
            self.__writeToLog('ERROR: Mini-batch size must be an integer.\n')
            return
        if miniBatchSize > 30000:
            self.__writeToLog('ERROR: Mini-batch size must be less than 30,000.\n')
            return
        
        try:
            inputSize = int(self.totalSizeInput.get())
            self.trainer.setInputSize(inputSize)
        except:
            self.__writeToLog('ERROR: Total images to use size must be an integer.\n')
            return
        if inputSize > 60000:
            self.__writeToLog('ERROR: Total images to use size must be less than 60,000.\n')
            return
        if inputSize < miniBatchSize:
            self.__writeToLog('ERROR: Total images to use must be greater than mini-batch size.\n')
            return
        
        try:
            gradientScaleFactor = float(self.scaleFactorInput.get())
            self.trainer.setGradientScaleFactor(gradientScaleFactor)
        except:
            self.__writeToLog('ERROR: Gradient descent scale-factor size must be a real number.\n')
            return
        if not gradientScaleFactor > 0:
            self.__writeToLog('ERROR: Gradient descent scale-factor size must be positive.\n')
            return
        
        self.__writeToLog('Verifying back-propagation trainer...\n')
        self.__writeToLog('Network structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] + '\n')
        self.__writeToLog('Mini-batch size is {}\n'.format(miniBatchSize))
        self.__writeToLog('Total images to use is {}\n'.format(inputSize))
        self.__writeToLog('Gradient descent scale-factor is {}\n'.format(gradientScaleFactor))
        self.__writeToLog('Verifying back-propagation trainer...done.\n\n')
        
        if not self.trainer.checkMNISTload():
            self.__writeToLog('Loading MNIST database to memory...')
            self.__updateTrainingProgressBar(10)
            self.trainer.loadMNIST()
            self.__writeToLog('done.\n')
            self.__updateTrainingProgressBar(100)
            time.sleep(0.2)
            self.trainingProgressBar['value'] = 0
        else:
            self.__writeToLog('MNIST database already loaded into memory.\n')
    
    def __updateTrainingProgressBar(self, progress):
        self.trainingProgressBar['value'] = progress
        root.update()
    
    def __updateEvaluateProgressBar(self, progress):
        self.evaluateProgressBar['value'] = progress
        root.update()
    
    def __evaluateNetwork(self):
        self.__clearLog()
        if self.trainer.getNetwork().getStructure() == []:
            self.__writeToLog('ERROR: No network to evaluate, intialise or load from file.\n')
            return

        self.__confusionMatrix = np.random.randint(0,10000,(10,10))
        self.__drawConfusionMatrix()
    
    def __trainAndEvaluateNetwork(self):
        self.__trainNetwork()
        self.__evaluateNetwork()
    
    def __writeToLog(self, message):
        self.messageLog.configure(state='normal')
        self.messageLog.insert(tk.END, message)
        self.messageLog.configure(state='disabled')
        self.messageLog.yview(tk.END)
        
    def __clearLog(self):
        self.messageLog.configure(state='normal')
        self.messageLog.delete(1.0, tk.END)
        self.messageLog.configure(state='disabled')
    
    def __initialiseNetwork(self):
        #try:
        hiddenLayersStr = re.findall('[0-9]+', self.structureInput.get())
        structure = [28*28]
        for layerStr in hiddenLayersStr:
            structure.append(int(layerStr))
        structure.append(10)
        seed = []
        if self.seedCheckVar.get() == 0:
            try:
                seed.append(int(self.seedInput.get()))
            except:
                self.__writeToLog('Random seed must be an integer, ignoring entered value.\n')
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