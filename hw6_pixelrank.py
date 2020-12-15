"""
Matt Russell
Motivation: AMATH584 HW6-Midterm2
12/14/2020

More direct script to examine effect of regularization on LASSO in order to rank
pixel importance
"""
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

file_ImageTrain = 'train-images-idx3-ubyte'
file_LabelTrain = 'train-labels-idx1-ubyte'
file_ImageTest = 't10k-images-idx3-ubyte'
file_LabelTest = 't10k-labels-idx1-ubyte'

ImageTrain_full = idx2numpy.convert_from_file(file_ImageTrain)
LabelTrain_full = idx2numpy.convert_from_file(file_LabelTrain)

numSamples = 50
numPixels = ImageTrain_full.shape[1] # images are square and isotropic
TrainSamples = ImageTrain_full.shape[0]

ImageTrain = np.empty((numSamples,numPixels,numPixels))
LabelTrain = np.empty((numSamples,1), dtype=int)

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

ImageTrain = ImageTrain.reshape((ImageTrain.shape[0],ImageTrain.shape[1]*ImageTrain.shape[2])) # gives under-determined A

## LASSO for several different regularizations
lam = np.array([0.1,0.25,0.5,0.75,1.0,2.0])
X_regLasso = np.zeros((lam.shape[0],LabelTrain.shape[1],ImageTrain.shape[1])) #(numRegularizations,numLabels,numPixels)

print(np.shape(X_regLasso))

for l in np.arange(lam.shape[0]):
    regLasso = linear_model.Lasso(alpha=lam[l],max_iter = 2000)
    regLasso.fit(ImageTrain,LabelTrain)
    X_regLasso[l,:,:] = regLasso.coef_
    print(regLasso.n_iter_)

"""
Find most important pixels
Take the Top Ten for each label taken from the average of weightings for all regularizations
"""
X_Average = np.zeros((X_regLasso.shape[1],X_regLasso.shape[2]))
for k in np.arange(X_regLasso.shape[1]): #numLabels
    for m in np.arange(X_regLasso.shape[2]): #numPixels
        X_Average[k,m] = np.average(X_regLasso[:,k,m])

print(np.shape(X_Average))

LabelOne_TopTen = np.zeros((2,10))
LabelTwo_TopTen = np.zeros((2,10))
LabelThree_TopTen = np.zeros((2,10))
LabelFour_TopTen = np.zeros((2,10))
LabelFive_TopTen = np.zeros((2,10))
LabelSix_TopTen = np.zeros((2,10))
LabelSeven_TopTen = np.zeros((2,10))
LabelEight_TopTen = np.zeros((2,10))
LabelNine_TopTen = np.zeros((2,10))
LabelZero_TopTen = np.zeros((2,10))





"""
Plotting
"""
figLasso, axs = plt.subplots(2,3)

ax = axs[0,0]
c = ax.pcolor(X_regLasso[0,:,:]/np.amax(X_regLasso[0,:,:]))
ax.set_title('Lambda = %1f' %lam[0])
figLasso.colorbar(c, ax=ax)

ax = axs[0,1]
c = ax.pcolor(X_regLasso[1,:,:]/np.amax(X_regLasso[1,:,:]))
ax.set_title('Lambda = %1f' %lam[1])
figLasso.colorbar(c, ax=ax)

ax = axs[0,2]
c = ax.pcolor(X_regLasso[2,:,:]/np.amax(X_regLasso[2,:,:]))
ax.set_title('Lambda = %1f' %lam[2])
figLasso.colorbar(c, ax=ax)

ax = axs[1,0]
c = ax.pcolor(X_regLasso[3,:,:]/np.amax(X_regLasso[3,:,:]))
ax.set_title('Lambda = %1f' %lam[3])
figLasso.colorbar(c, ax=ax)

ax = axs[1,1]
c = ax.pcolor(X_regLasso[4,:,:]/np.amax(X_regLasso[4,:,:]))
ax.set_title('Lambda = %1f' %lam[4])
figLasso.colorbar(c, ax=ax)

ax = axs[1,2]
c = ax.pcolor(X_regLasso[5,:,:]/np.amax(X_regLasso[5,:,:]))
ax.set_title('Lambda = %1f' %lam[5])
figLasso.colorbar(c, ax=ax)

figLasso.suptitle('Loadings for LASSO with different regularizations')
figLasso.tight_layout()

figAverage = plt.figure(2)
plt.pcolor(X_Average)

plt.show()
