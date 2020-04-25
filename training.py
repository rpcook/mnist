# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import ttk
from struct import pack
from random import random
import backpropagation as bp

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
        self.seedInput.grid(row=1, column=1, sticky='W')
        self.seedCheckVar = tk.IntVar()
        tk.Checkbutton(text='Randomise', variable=self.seedCheckVar, command=self.__randomiseCheck).grid(row=2, column=1, sticky='W')
        
        tk.Button(text='Initialise Network').grid(row=3, column=0)
        tk.Button(text='Load Network').grid(row=3, column=1)
        
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
        
        self.messageLog = scrolledtext.ScrolledText(height=5, width=40, wrap=tk.WORD)
        self.messageLog.grid(row=7, column=0, columnspan=2)
        
        self.trainingProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.trainingProgressBar.grid(row=8, column=0, columnspan=2)
        
        tk.Button(text='Save Network').grid(row=9, column=0)
        tk.Button(text='Save Network & Log').grid(row=9, column=1)
        
        tk.Button(text='Evaluate Network Performance').grid(row=3, column=2)

        self.evaluateProgressBar = ttk.Progressbar(orient = tk.HORIZONTAL, 
              length = 200, mode = 'determinate')
        self.evaluateProgressBar.grid(row=4, column=2)
        
        self.pxSize = 30
        self.confusionCanvas = tk.Canvas(width=11*self.pxSize, height=11*self.pxSize)
        self.confusionCanvas.grid(row=5, column=2, rowspan=4)
        
        self.performanceLabelContent = tk.StringVar()
        self.performanceLabelContent.set('Network accuracy: #')
        self.performanceLabel = tk.Label(textvariable=self.performanceLabelContent)
        self.performanceLabel.grid(row=9, column=2)
        
        self.trainer = bp.trainer()
    
    def __randomiseCheck(self):
        if self.seedCheckVar.get() == 0:
            self.seedInput.configure(state='normal')
        else:
            self.seedInput.configure(state='disabled')
    
    def __createNetwork(self):
        pass
    
    def __testButton(self):
        file = filedialog.asksaveasfile(filetypes=(('nn files', '\*.nn'),))
        if file is None:
            return
        trainer = bp.trainer()
        trainer.initialiseNetwork([28*28, 16, 16, 10])
        self.__saveNetwork(trainer.getNetwork(), file.name)
    
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