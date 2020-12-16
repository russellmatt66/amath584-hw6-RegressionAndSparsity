"""
Matt Russell
Motivation: AMATH584 HW6-Midterm2
Date: 12/13/20

Basic script to extract MNIST data and perform some routine regression on it
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

## Least Squares Regression
regLS = linear_model.LinearRegression().fit(ImageTrain,LabelTrain)
x_regLS = regLS.coef_ #coef_ attribute contains the loadings (particular solution)

## Ridge Regression
lam2 = 1.0
regRidge = linear_model.Ridge(alpha=lam2)
regRidge.fit(ImageTrain,LabelTrain)
x_regRidge = regRidge.coef_

## LASSO -
lam1 = 1.0
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
figReg, axs = plt.subplots(2,2)

ax = axs[0,0]
c = ax.pcolor(x_pseudoinv.T/np.amax(x_pseudoinv))
ax.set_title('Pseudo-Inverse Weights')
ax.set_xlabel('Pixel')
ax.set_ylabel('Particular Label')
figReg.colorbar(c, ax=ax)

ax = axs[0,1]
c = ax.pcolor(x_regLS/np.amax(x_regLS))
ax.set_title('Least-Squares Regression Weights')
ax.set_xlabel('Pixel')
ax.set_ylabel('Particular Label')
figReg.colorbar(c, ax=ax)

ax = axs[1,0]
c = ax.pcolor(x_regRidge/np.amax(x_regRidge))
ax.set_title('Ridge Regression Weights')
ax.set_xlabel('Pixel')
ax.set_ylabel('Particular Label')
figReg.colorbar(c, ax=ax)

ax = axs[1,1]
c = ax.pcolor(x_regLasso/np.amax(x_regLasso))
ax.set_title('LASSO Weights')
ax.set_xlabel('Pixel')
ax.set_ylabel('Particular Label')
figReg.colorbar(c, ax=ax)

figReg.suptitle('Normalized Loadings for Several Different Regression Methods')
figReg.tight_layout()
plt.show()
