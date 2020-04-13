# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:00:58 2020

@author: Rob.Cook
"""

import tkinter as tk
from struct import pack
from random import random

class testingGUI:
    def __init__(self, master):
        self.master = master
        master.title('MNIST Neural Network Training Interface')
        
        self.prototypeTrainingButton = tk.Button(text='Save Random NN', command=self.__saveRandomNN)
        self.prototypeTrainingButton.grid(row=1)
        
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