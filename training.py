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
        
        tk.Label(text='Gradient descent scale-factor:').grid(row=5, column=0)
        self.scaleFactorInput = tk.Entry()
        self.scaleFactorInput.insert(0, '1.0')
        self.scaleFactorInput.grid(row=5, column=1, sticky='W')
        
        tk.Button(text='Train Network', command=self.__trainNetwork).grid(row=6, column=0)
        tk.Button(text='Train Network & Evaluate Performance', command=self.__trainAndEvaluateNetwork).grid(row=6, column=1)
        
        self.trainingProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.trainingProgressBar.grid(row=7, column=0, columnspan=2, sticky='EW')
        
        tk.Button(text='Save Network', command=self.__saveButtonHandler).grid(row=8, column=0)
        tk.Button(text='Save Network & Log').grid(row=8, column=1)
        
        self.messageLog = scrolledtext.ScrolledText(height=5, width=70, wrap=tk.WORD, state='disabled')
        self.messageLog.grid(row=9, column=0, columnspan=2)
        
        ttk.Separator(orient=tk.VERTICAL).grid(row=0, column=2, rowspan=10, sticky='NS')
        
        tk.Button(text='Evaluate Network\nPerformance', command=self.__evaluateNetwork).grid(row=0, column=3, rowspan=2)

        self.evaluateProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.evaluateProgressBar.grid(row=0, column=4)
        
        self.performanceLabelContent = tk.StringVar()
        self.performanceLabelContent.set('Overall network accuracy: #')
        self.performanceLabel = tk.Label(textvariable=self.performanceLabelContent)
        self.performanceLabel.grid(row=1, column=4)
        
        self.pxSize = 30
        self.confusionCanvas = tk.Canvas(width=11*self.pxSize, height=11*self.pxSize, bg='#000040')
        self.confusionCanvas.grid(row=2, column=3, rowspan=8, columnspan=2)
        
        self.trainer = bp.trainer()
        self.networkPerformance = np.zeros((10,10))
        
        self.__drawConfusionMatrix()
    
    def __drawConfusionMatrix(self):
        pass
    
    def __trainNetwork(self):
        self.__clearLog()
        try:
            miniBatchSize = int(self.batchSizeInput.get())
        except:
            self.__writeToLog('ERROR: Mini-batch size must be an integer.\n')
            return
        if self.trainer.getNetwork().getStructure() == []:
            self.__writeToLog('ERROR: No network to train, intialise or load from file.\n')
            return
        self.__writeToLog('Training network...\n')
        self.__writeToLog('Network structure: ' + str(self.trainer.getNetwork().getStructure())[1:-1] + '\n')
        self.__writeToLog('Mini-batch size is {}\n'.format(miniBatchSize))
        if not self.trainer.checkMNISTload():
            self.__writeToLog('Loading MNIST database to memory...')
            self.trainingProgressBar['value'] = 50
            root.update()
            self.trainer.loadMNIST()
            self.trainingProgressBar['value'] = 100
            self.__writeToLog('done.\n')
            root.update()
            time.sleep(0.2)
            self.trainingProgressBar['value'] = 0
        pass
    
    def __evaluateNetwork(self):
        print('evaluate')
        pass
    
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