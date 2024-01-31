import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random

directory_base_address = 'E:/OneDrive - TUNI.fi/Tampere University (MSc in CS-DS)/Year 2/Period 1/1. Data.ML.100 Introduction to Pattern Recognition and Machine Learning/Exercises/Exercise_3/cifar-10-python\cifar-10-batches-py'

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle(directory_base_address+'/data_batch_1')
datadict_test = unpickle(directory_base_address+'/test_batch')

X = datadict["data"]
Y = datadict["labels"]

X_test= datadict_test["data"]
Y_test = datadict_test["labels"]

#print(X.shape)

labeldict = unpickle(directory_base_address+'/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
Y = np.array(Y)

X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
Y_test = np.array(Y_test)

for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)


# Evaluation Function
def class_acc(pred,gt):
    
    correctly_classified = 0
    total_sample = len(pred)
    
    for i in range(total_sample):
        if pred[i] == gt[i]:
            correctly_classified=correctly_classified+1
            
    #print(correctly_classified)
    accuracy = (correctly_classified/total_sample)*100
    print("Model Accuracy: ", accuracy)
    

# Random Classifier
def cifar10_classifier_random(X):
    
    label = np.random.randint(0,9,X.shape[0])
    class_acc(label, Y_test)

cifar10_classifier_random(X_test)

# 1-NN Classifier
def cifar10_classifier_1nn(X,trdata,trlabels):
    labels_1nn = []
    for i in range(X.shape[0]):
        distances = []
        for j in range(trdata.shape[0]):
            distances.append(np.sqrt(np.sum(np.square((trdata[j]-X[i])))))
        
        min_dis = min(distances)
        index = distances.index(min_dis)
        label = trlabels[index]
        labels_1nn.append(label)
       
        """
        #Plottting thr figure
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[label]} (num {label})")
        plt.pause(1)
    """

    class_acc(np.array(labels_1nn), Y_test)

cifar10_classifier_1nn(X_test,X,Y) 


