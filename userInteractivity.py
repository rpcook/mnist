# -*- coding: utf-8 -*-
import tkinter as tk
import numpy as np

class elements:
    def __init__(self, kwargs):
        if 'rootWindow' in kwargs and 'progressBarWidget' in kwargs:
            self.rootWindow = kwargs['rootWindow']
            self.progressBarWidget = kwargs['progressBarWidget']
            self.progressBarAvailable = True
        else:
            self.progressBarAvailable = False
            
        if 'rootWindow' in kwargs and 'messageLog' in kwargs:
            self.rootWindow = kwargs['rootWindow']
            self.messageLog = kwargs['messageLog']
            self.messageLogAvailable = True
        else:
            self.messageLogAvailable = False
            
        if 'rootWindow' in kwargs and 'graphCanvas' in kwargs:
            self.rootWindow = kwargs['rootWindow']
            self.graphCanvas = kwargs['graphCanvas']
            self.graphAvailable = True
        else:
            self.graphAvailable = False

    def drawGraphs(self, lossHistoryTrainer, lossHistoryValidation, errorHistory):
        if self.graphAvailable:
            gc = self.graphCanvas
            lht = lossHistoryTrainer
            lhv = lossHistoryValidation
            gc.delete('all')
            gc.create_text(7,75,angle=90,text='Training loss',fill='blue')
            gc.create_text(20,75,angle=90,text='Validation loss',fill='green')
            gc.create_text(31,75,angle=90,text='Error',fill='orange')
            
            boldLine = '#aaaaaa'
            softLine = '#dddddd'
            if len(lht) < 2:
                numMajorGrids = 1
            else:
                if min(*errorHistory) == 0:
                    errorHistoryNoZeros = []
                    minimumElsewhere = min(*lht,*lhv)
                    for entry in errorHistory:
                        if entry > 0:
                            errorHistoryNoZeros.append(entry)
                        else:
                            errorHistoryNoZeros.append(10**np.floor(np.log10(minimumElsewhere)))
                else:
                    errorHistoryNoZeros = errorHistory
                numMajorGrids = int(np.ceil(np.log10(10/min(*lht,*lhv,*errorHistoryNoZeros))))
            
            gc.create_line(40,8,40,145)
            for i in range(numMajorGrids+1):
                gc.create_line(41, 8+i*(136/numMajorGrids), 500, 8+i*(136/numMajorGrids), fill=boldLine)
            for i in range(numMajorGrids):
                for j in range(2,10):
                    yPos = 8+(i+1-np.log10(j))*(136/numMajorGrids)
                    gc.create_line(41, yPos, 500, yPos, fill=softLine)
            
            if min(len(lht), len(lhv), len(errorHistory)) != max(len(lht), len(lhv), len(errorHistory)):
                return
            
            if len(lht) > 1:
                maxPoints = 600
                interval = int(np.ceil(len(lht)/maxPoints))
                for i in range(0, len(lht)-interval, interval):
                    yPosStart = 8 + 136*((1 - np.log10(lhv[i])) / numMajorGrids)
                    yPosStop  = 8 + 136*((1 - np.log10(lhv[i+interval])) / numMajorGrids)
                    gc.create_line(41+i*(460/(len(lhv)-1)), yPosStart, 41+(i+interval)*(460/(len(lhv)-1)), yPosStop, fill='green')
                    yPosStart = 8 + 136*((1 - np.log10(lht[i])) / numMajorGrids)
                    yPosStop  = 8 + 136*((1 - np.log10(lht[i+interval])) / numMajorGrids)
                    gc.create_line(41+i*(460/(len(lht)-1)), yPosStart, 41+(i+interval)*(460/(len(lht)-1)), yPosStop, fill='blue')
                    yPosStart = 8 + 136*((1 - np.log10(errorHistoryNoZeros[i])) / numMajorGrids)
                    yPosStop  = 8 + 136*((1 - np.log10(errorHistoryNoZeros[i+interval])) / numMajorGrids)
                    gc.create_line(41+i*(460/(len(errorHistoryNoZeros)-1)), yPosStart, 41+(i+interval)*(460/(len(errorHistoryNoZeros)-1)), yPosStop, fill='orange')
                if i > maxPoints:
                    i = len(lht) - 2
                    yPosStart = 8 + 136*((1 - np.log10(lhv[i])) / numMajorGrids)
                    yPosStop  = 8 + 136*((1 - np.log10(lhv[i+1])) / numMajorGrids)
                    gc.create_line(41+(i-interval+2)*(460/(len(lhv)-1)), yPosStart, 41+(i+1)*(460/(len(lhv)-1)), yPosStop, fill='green')
                    yPosStart = 8 + 136*((1 - np.log10(lht[i])) / numMajorGrids)
                    yPosStop  = 8 + 136*((1 - np.log10(lht[i+1])) / numMajorGrids)
                    gc.create_line(41+(i-interval+2)*(460/(len(lht)-1)), yPosStart, 41+(i+1)*(460/(len(lht)-1)), yPosStop, fill='blue')
                    yPosStart = 8 + 136*((1 - np.log10(errorHistoryNoZeros[i])) / numMajorGrids)
                    yPosStop  = 8 + 136*((1 - np.log10(errorHistoryNoZeros[i+1])) / numMajorGrids)
                    gc.create_line(41+(i-interval+2)*(460/(len(errorHistoryNoZeros)-1)), yPosStart, 41+(i+1)*(460/(len(errorHistoryNoZeros)-1)), yPosStop, fill='orange')
            
            self.rootWindow.update()

    def writeToLog(self, message):
        if self.messageLogAvailable:
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
            self.rootWindow.update()

    def updateProgressBar(self, progress):
        if self.progressBarAvailable:
            self.progressBarWidget['value'] = progress   
            self.rootWindow.update()
