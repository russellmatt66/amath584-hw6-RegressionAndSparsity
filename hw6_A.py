"""
Matt Russell
Motivation: AMATH584 HW6-Midterm2
Date: 12/13/20
"""
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

# Extract MNIST data and turn into numpy arrays
file_ImageTrain = 'train-images-idx3-ubyte'
file_LabelTrain = 'train-labels-idx1-ubyte'
file_ImageTest = 't10k-images-idx3-ubyte'
file_LabelTest = 't10k-labels-idx1-ubyte'

ImageTrain_full = idx2numpy.convert_from_file(file_ImageTrain)
LabelTrain_full = idx2numpy.convert_from_file(file_LabelTrain)
ImageTest_full = idx2numpy.convert_from_file(file_ImageTest)
LabelTest_full = idx2numpy.convert_from_file(file_LabelTest)

print(np.shape(ImageTrain_full))
print(np.shape(LabelTrain_full))
print(np.shape(ImageTest_full))
print(np.shape(LabelTest_full))

# Grab a number of random images and labels
numSamples = 50
numPixels = ImageTrain_full.shape[1] # images are square and isotropic
TrainSamples = ImageTrain_full.shape[0]
TestSamples = ImageTest_full.shape[0]

ImageTrain = np.empty((numSamples,numPixels,numPixels))
LabelTrain = np.empty((numSamples,1), dtype=int)
ImageTest = np.empty((numSamples,numPixels,numPixels))
LabelTest = np.empty((numSamples,1), dtype=int)

for j in np.arange(numSamples):
    randSample = np.random.randint(0,TrainSamples-1)
    ImageTrain[j,:,:] = ImageTrain_full[randSample,:,:]
    LabelTrain[j] = LabelTrain_full[randSample]

LabelTrain_temp = np.zeros((LabelTrain.shape[0],10))

# Transform MNIST labels to our labels
for k in np.arange(LabelTrain.shape[0]):
    # print(k)
    # print(LabelTrain[k])
    LabelTrain_temp[k,(LabelTrain[k]+9) % 10] = 1

LabelTrain = LabelTrain_temp

print(np.shape(ImageTrain))
print(np.shape(LabelTrain))
print(LabelTrain)
#print(np.shape(ImageTest_full))
#print(np.shape(LabelTest_full))

ImageTrain = ImageTrain.reshape((ImageTrain.shape[0],ImageTrain.shape[1]*ImageTrain.shape[2])) # gives under-determined A
print(np.shape(ImageTrain))

## Pseudo-Inverse
x_pseudoinv = np.matmul(np.linalg.pinv(ImageTrain),LabelTrain)
print(np.shape(x_pseudoinv))

lam1 = 1.0
lam2 = 1.0
## Least Squares Regression
regLS = linear_model.LinearRegression().fit(ImageTrain,LabelTrain)
x_regLS = regLS.coef_

## Ridge Regression
regRidge = linear_model.Ridge(alpha=lam2)
regRidge.fit(ImageTrain,LabelTrain)
x_regRidge = regRidge.coef_

## LASSO -
regLasso = linear_model.Lasso(alpha=lam1,max_iter = 1000, tol = 0.0001)
regLasso.fit(ImageTrain,LabelTrain)
x_regLasso = regLasso.coef_
print(regLasso.n_iter_)

print(np.shape(x_regLS))
print(np.shape(x_regRidge))
print(np.shape(x_regLasso))

"""
Plotting
"""
