#Importing necessary libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from random import random

#defining base directory
directory_base_address = 'E:/OneDrive - TUNI.fi/Tampere University (MSc in CS-DS)/Year 2/Period 1/1. Data.ML.100 Introduction to Pattern Recognition and Machine Learning/Exercises/Exercise_3/cifar-10-python\cifar-10-batches-py'

#function to read files
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# declare empty lists to store training data from different files
X =[]
Y =[]

# reading training data
for i in range(1,6):
    datadict = unpickle(directory_base_address+'/data_batch_'+str(i))
    X.append(datadict["data"])
    Y.append(datadict["labels"])

# reading test data
datadict_test = unpickle(directory_base_address+'/test_batch')

# merging all the training data into one    
X_train = np.concatenate(X, axis=0 )
Y_train = Y[0]+Y[1]+Y[2]+Y[3]+Y[4]

X_test= datadict_test["data"]
Y_test = datadict_test["labels"]

# reading labels data
labeldict = unpickle(directory_base_address+'/batches.meta')
label_names = labeldict["label_names"]

# transposing the data's into image
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
Y_train = np.array(Y_train)

X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
Y_test = np.array(Y_test)

"""
# ploting images
for i in range(X_train.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X_train[i])
        plt.title(f"Image {i} label={label_names[Y_train[i]]} (num {Y_train[i]})")
        plt.pause(1)

""" 

# 2. CIFAR-10:  Evaluation Function
def class_acc(pred,gt):
    
    correctly_classified = 0
    total_sample = len(pred)
    
    for i in range(total_sample):
        if pred[i] == gt[i]:
            correctly_classified=correctly_classified+1
            
    #print(correctly_classified)
    accuracy = (correctly_classified/total_sample)*100
    print("Model Accuracy: ", accuracy)
    

#3. CIFAR-10: Random Classifier
def cifar10_classifier_random(X):
    
    label = np.random.randint(0,9,X.shape[0])
    class_acc(label, Y_test)

cifar10_classifier_random(X_test)

#4. CIFAR-10: 1-NN Classifier
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
       
    
        #Plottting the figure
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[label]} (num {label})")
        plt.pause(1)


    class_acc(np.array(labels_1nn), Y_test)

cifar10_classifier_1nn(X_test,X_train,Y_train) 


