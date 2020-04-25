# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import messagebox
from struct import pack
from random import random
import backpropagation as bp
import neuralnetwork as nn
from struct import unpack
import numpy as np

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
        
        tk.Button(text='Initialise Network').grid(row=3, column=0)
        tk.Button(text='Load Network', command=self.__loadNetwork).grid(row=3, column=1)
        
        tk.Label(text='Mini-batch size:').grid(row=4, column=0)
        self.batchSizeInput = tk.Entry()
        self.batchSizeInput.insert(0, '100')
        self.batchSizeInput.grid(row=4, column=1, sticky='W')
        
        tk.Label(text='Gradient descent scale-factor:').grid(row=5, column=0)
        self.scaleFactorInput = tk.Entry()
        self.scaleFactorInput.insert(0, '1.0')
        self.scaleFactorInput.grid(row=5, column=1, sticky='W')
        
        tk.Button(text='Train Network').grid(row=6, column=0)
        tk.Button(text='Train Network & Test Performance').grid(row=6, column=1)
        
        self.messageLog = scrolledtext.ScrolledText(height=5, width=40, wrap=tk.WORD, state='disabled')
        self.messageLog.grid(row=7, column=0, columnspan=2)
        
        self.trainingProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.trainingProgressBar.grid(row=8, column=0, columnspan=2)
        
        tk.Button(text='Save Network', command=self.__saveButtonHandler).grid(row=9, column=0)
        tk.Button(text='Save Network & Log').grid(row=9, column=1)
        
        tk.Button(text='Evaluate Network Performance').grid(row=3, column=2)

        self.evaluateProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.evaluateProgressBar.grid(row=4, column=2)
        
        self.pxSize = 30
        self.confusionCanvas = tk.Canvas(width=11*self.pxSize, height=11*self.pxSize)
        self.confusionCanvas.grid(row=5, column=2, rowspan=4)
        
        self.performanceLabelContent = tk.StringVar()
        self.performanceLabelContent.set('Overall network accuracy: #')
        self.performanceLabel = tk.Label(textvariable=self.performanceLabelContent)
        self.performanceLabel.grid(row=9, column=2)
        
        self.trainer = bp.trainer()
    
    def __writeToLog(self, message):
        self.messageLog.configure(state='normal')
        self.messageLog.insert(tk.END, message)
        self.messageLog.configure(state='disabled')
        self.messageLog.yview(tk.END)
        
    def __clearLog(self):
        self.messageLog.configure(state='normal')
        self.messageLog.delete(1.0, tk.END)
        self.messageLog.configure(state='disabled')
    
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
    
    def __createNetwork(self):
        pass
    
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

    def __saveRandomNN(self):
        with open('randomNetwork.nn', 'wb') as f:
            f.write(pack('B', 4))
            f.write(pack('<4H', 28*28, 16, 16, 10))
            for a in range(16): # weights for 1st layer
                for b in range(28*28):
                    f.write(pack('<f', 2*random()-1))
            for a in range(16): # biases for 1st layer
                f.write(pack('<f', 2*random()-1))
                
            for a in range(16): # weights for 2nd layer
                for b in range(16):
                    f.write(pack('<f', 2*random()-1))
            for a in range(16): # biases for 2nd layer
                f.write(pack('<f', 2*random()-1))
                
            for a in range(10): # weights for last layer
                for b in range(16):
                    f.write(pack('<f', 2*random()-1))
            for a in range(10): # biases for last layer
                f.write(pack('<f', 2*random()-1))
        
root = tk.Tk()
g = testingGUI(root)
root.mainloop()