#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:49:29 2017

@author: anmo
"""

import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix

class confusionmatrix:
    
    # Rows correspond to true labels
    # Columns correspond to predicted labels
    
    def __init__(self, numClasses, labels=None):
        self.numClasses = numClasses
        self.confMat = np.zeros(shape=(numClasses,numClasses))
        self.labels = labels
        
    def __str__(self):
        numExamples = np.sum(self.confMat)
        width = np.maximum(5, np.ceil(np.log10(numExamples)))

        if self.labels is not None:
            min_label_len = np.min([len(l) for l in self.labels])
            label_abv_len = min_label_len
            for k in range(0, min_label_len):
                if len(np.unique([label[0:k] for label in self.labels])) == len(self.labels):
                    label_abv_len = k
                    break
            labels_abv = [label[0:label_abv_len] for label in self.labels]
        else:
            label_abv_len = 0
            labels_abv = ['' for i in range(self.numClasses)]
        

        obs_str = ('Observed'+''.join([' ' for _ in range(10000)]))[0:self.numClasses]
        # Predicted header
        str_out = '  ' + ' '*(label_abv_len+1) +  '| ' + '{:{width}s}'.format('Predicted',width=int((1+width)*self.numClasses)) + '| \r\n'
        if self.labels is not None:
            str_out += '  ' + ' '*(label_abv_len+1) +  '| ' + '{:{width}s}'.format(' '*int(width-label_abv_len) + (' '*int(width-label_abv_len+1)).join(labels_abv), width=int((1+width)*self.numClasses)) + '| \r\n'
        # Line seperator
        str_out += '--' + '-'*(label_abv_len+1) +  '+' + ''.join(['-' for _ in range(int((1+width)*self.numClasses))]) + '-+-' + '-----' + '\r\n'
        # Create confusion matrix str
        recalls = self.recall()
        precisions = self.precision()
        # Loop through each row (observed class), and then column (predicted class)
        for row, obs_char, label_abv, recall in zip(self.confMat, obs_str, labels_abv, recalls):
            str_out += obs_char + ' ' + label_abv +  ' |'
            for col in row:
                str_out += ' {:>{width}d}'.format(int(col), width=int(width))
            # Add class recall to end of row
            str_out += ' | ' + '{:5.3f}'.format(recall) + '\r\n'
        # Line seperator
        str_out += '--' + '-'*(label_abv_len+1) +  '+' + ''.join(['-' for _ in range(int((1+width)*self.numClasses))]) + '-+-' + '-----' + '\r\n'
        # Precision at end of each column
        str_out += '  ' + ' '*(label_abv_len+1) +  '|'
        for precision in precisions:
            str_out += ' ' + '{:>{width}.3f}'.format(precision, width=int(width))
        # Accuracy
        str_out += ' | ' + '{:5.3f}'.format(self.accuracy()) + '\r\n'
        return str_out
    
    def Append(self, trueLabels, predictedLabels, useOldCode=False):

        if (not isinstance(trueLabels,(list,np.ndarray))):
            trueLabels = [trueLabels]
        if (not isinstance(predictedLabels,(list,np.ndarray))):
            predictedLabels = [predictedLabels]

        # Convert to numpy arrays
        trueLabels = np.asarray(trueLabels)
        predictedLabels = np.asarray(predictedLabels).reshape(trueLabels.shape)
        
        # Size of true labels and predicted labels must be the same
        assert(trueLabels.shape == predictedLabels.shape)

        
        # Setup temp confusion matrix
        if (useOldCode):
            # Setup temp confusion matrix
            confMat_tmp = np.zeros(shape=(self.numClasses,self.numClasses))
            for t in range(self.numClasses):
                # Get index of all true labels belonging to class 't'
                tIdx = np.where(np.in1d(trueLabels, t))
                for p in range(self.numClasses):
                    # Get index of all predictions belonging to class 'p'
                    pIdx = np.where(np.in1d(predictedLabels, p))
                    # Find the indicies where both true label belongs to 't' and prediction belongs to 'p'
                    intersection = np.intersect1d(tIdx,pIdx)
                    # Store 
                    confMat_tmp[t][p] = len(intersection)

#            # Alternative #2
#            confMat_tmp = np.zeros(shape=(self.numClasses,self.numClasses))  
#            for idx,trueLabel in enumerate(trueLabels):
#                #print(trueLabels[idx])
#                #print(predictedLabels[idx])
#                confMat_tmp[trueLabels[idx]][predictedLabels[idx]] += 1
                
        else: # Use skilearns method for creating confusion matrix instead
            # Convert label arrays to vectors
            trueLabels = np.reshape(trueLabels, (np.prod(trueLabels.shape)))
            predictedLabels = np.reshape(predictedLabels, (np.prod(predictedLabels.shape)))
            
            confMat_tmp = confusion_matrix(trueLabels, predictedLabels, range(self.numClasses))
        # Add temp confusion matrix to instance confusion matrix
        self.confMat = self.confMat + confMat_tmp
        return self
    
    def Reset(self, confMat_new=None):
        if confMat_new is None:
            confMat_new = np.zeros(shape=(self.numClasses, self.numClasses))
        confMat_new = np.asarray(confMat_new)
        if self.confMat.shape != confMat_new.shape:
            raise ValueError('Shape of new confusion matrix must match old of old confusion matrix when resetting.')
        # Set set all elemants in instance confusion matrix to 0
        self.confMat = confMat_new
        return self
    
    def Save(self, file, fileFormat=None):
        # Save the confusion matrix into a binary file in Numpy *.npy format
        
        # Determine fileformat is not specified
        if (fileFormat == None):
            filename, fileExtension = os.path.splitext(file)
            fileExtension = fileExtension[1:]
            if fileExtension in {'csv','txt','dat'}:
                fileFormat = 'csv'
            elif fileExtension in {'npy'}:
                fileFormat = 'npy'
            elif fileExtension in {'npz'}:
                fileFormat = 'npz'
            else:
                raise ValueError('Could not determine the appropriate file format from the specified file.')

        # Save the confusion matrix using the appropriate numpy saver
        if (fileFormat == 'csv'):
            np.savetxt(file, self.confMat,delimiter=',',fmt='%d')
        elif (fileFormat == 'npy'):
            np.save(file, self.confMat, allow_pickle=False)
        elif (fileFormat == 'npz'):
            np.savez(file, confMat = self.confMat)
        else:
            raise ValueError('Unknown file format.')
        
    def Load(self, file, fileFormat=None):
        # Load a previously saved confusion matrix
        
        # Determine fileformat is not specified
        if (fileFormat == None):
            filename, fileExtension = os.path.splitext(file)
            fileExtension = fileExtension[1:]
            if fileExtension in {'csv','txt','dat'}:
                fileFormat = 'csv'
            elif fileExtension in {'npy'}:
                fileFormat = 'npy'
            elif fileExtension in {'npz'}:
                fileFormat = 'npz'
            else:
                raise ValueError('Could not determine the appropriate file format from the specified file.')

        # Load the confusion matrix using the appropriate numpy saver
        if (fileFormat == 'csv'):
            self.confMat = np.loadtxt(file, delimiter=',')
        elif (fileFormat == 'npy'):
            self.confMat = np.load(file, mmap_mode=None, allow_pickle=False)
        elif (fileFormat == 'npz'):
            data = np.load(file, mmap_mode=None, allow_pickle=False)
            self.confMat = data['confMat']
        else:
            raise ValueError('Unknown file format.')
        self.numClasses = self.confMat.shape[0]

    def MergeLabels(self, merge_list, new_labels=None):
        # NOTE: Experimental
        if merge_list is None:
            pass
        ConfMat_tmp = np.zeros((len(merge_list), self.numClasses))
        for g, group in enumerate(merge_list):
            group_row = np.zeros((1, self.numClasses))
            for member in group:
                group_row += self.confMat[member,:]
            ConfMat_tmp[g,:] = group_row

        ConfMat_new = np.zeros((len(merge_list), len(merge_list)))
        for g, group in enumerate(merge_list):
            group_column = np.zeros((1, len(merge_list)))
            for member in group:
                group_column += ConfMat_tmp[:,member]
            ConfMat_new[:, g] = group_column

        CMmerged = confusionmatrix(len(merge_list), labels=new_labels)
        CMmerged.Reset(ConfMat_new)
        return CMmerged
        
    ## METRICS ##
    def count(self):
        # Number of samples
        return np.sum(self.confMat, axis=(0,1))
    
    def trueCounts(self):
        # Number of true labels from each class
        return np.sum(self.confMat, axis=1)

    def predictedCounts(self):
        # Number of predicted labels from each class
        return np.sum(self.confMat, axis=0)
    
    def trueFrequency(self):
        # True relative frequency of each class
        TC = self.trueCounts()
        return TC / np.sum(TC)

    def predictedFrequency(self):
        # Predicted relative frequency of each class
        PC = self.predictedCounts()
        return PC / np.sum(PC)
        
    def truePositives(self):
        # Number class of interest classified as class of interest
        return np.diagonal(self.confMat)
    
    def falseNegatives(self):
        # Number of class of interest classified as other classes
        return self.trueCounts() - self.truePositives()

    def falsePositives(self):
        # Number of other classes predicted as class of interest
        return self.predictedCounts() - self.truePositives()
        
    def trueNegatives(self):
        # Number of other classes classified as another class than the class of interest
        return np.subtract(self.count(), self.truePositives() + self.falsePositives() + self.falseNegatives())
        
    def truePositiveRates(self):
        # True positive rate of each class
        P = self.trueCounts()
        TP = self.truePositives()
        TPR = np.divide(TP,P)
        return TPR
    
    def precision(self):
        # Precision of each class
        # Number of true positive predictions divided by the total number of positive predictions
        TP = self.truePositives()
        TPFP = self.predictedCounts()
        return np.divide(TP,TPFP)
    
    def recall(self):
        # Recall of each class.
        # Same as true positive rate. See truePositiveRates()
        return self.truePositiveRates()
        
    def accuracy(self):
        # Calculate accuracy
        # sum of true predictions divided by all predictions
        return np.sum(np.diagonal(self.confMat))/np.sum(self.confMat)
    
    def intersectionOverUnion(self):
        # Calculate intersection over union for each class
        # True prediction of a given class divided by the total number of predictions and true labels of that class
        iou = np.zeros(shape=(self.numClasses))
        for t in range(self.numClasses):
            union = (np.sum(self.confMat[:,t]) + np.sum(self.confMat[t,:]) - self.confMat[t,t])
            intersection = self.confMat[t,t]
            if (union == 0):
                iou[t] = 0
            else:   
                iou[t] = intersection / union
        return iou
    
    def jaccardIndex(self):
        # Same as intersection over union. See confusionmatrix.intersectionOverUnion()
        # Has a close relationship to Dice's coefficient
        return self.intersectionOverUnion()
        
    def fScore(self, beta=1):
        P = self.precision()
        R = self.recall()
        # return (1+beta*beta)*(P*R)/(beta*beta*P + R)
        return (1+beta*beta)*(P*R)/(beta*beta*P + R)
        
    def dicesCoefficient(self):
        # Same as F1-score and has a close relationship to Jaccard index
        J = self.jaccardIndex()
        return np.divide(2*J,1+J)

if (__name__ == '__main__'):
    CMat = confusionmatrix(2)
    CMat.Load(sys.argv[1])
    print(CMat)
