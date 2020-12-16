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

sizeLabels = 10
LabelTrain_temp = np.zeros((LabelTrain.shape[0],sizeLabels))

# Transform MNIST labels to our labels
for k in np.arange(LabelTrain.shape[0]):
    LabelTrain_temp[k,(LabelTrain[k]+9) % 10] = 1

LabelTrain = LabelTrain_temp

ImageTrain = ImageTrain.reshape((ImageTrain.shape[0],ImageTrain.shape[1]*ImageTrain.shape[2])) # gives under-determined A

## LASSO for several different regularizations
lam = np.array([0.1,0.25,0.5,0.75,1.0,2.0])
X_regLasso = np.zeros((lam.shape[0],LabelTrain.shape[1],ImageTrain.shape[1])) #(numRegularizations,sizeLabels,numPixels^2), interpretation of axis = 1 is ("1","2",...,"0")

print(np.shape(X_regLasso))

for l in np.arange(lam.shape[0]):
    regLasso = linear_model.Lasso(alpha=lam[l],max_iter = 2000)
    regLasso.fit(ImageTrain,LabelTrain)
    X_regLasso[l,:,:] = regLasso.coef_
    print(regLasso.n_iter_)

"""
Find most important pixels
Average the weightings for each pixel across all regularizations and then take the top 10
"""
X_Average = np.zeros((X_regLasso.shape[1],X_regLasso.shape[2]))
for k in np.arange(X_regLasso.shape[1]): #numLabels
    for m in np.arange(X_regLasso.shape[2]): #numPixels
        X_Average[k,m] = np.average(X_regLasso[:,k,m])

print(np.shape(X_Average))

n_important = 10
TopTen_ind = np.zeros((LabelTrain.shape[1],n_important),dtype=int) # axis = 0 represents each of the different labels with the same interpretation of ("1","2",..."0"), the indices of the pixels with the largest weightings for the respective label are stored in ascending order along axis = 1
for i in np.arange(X_Average.shape[0]):
    temp_array = X_Average[i,:]
    temp_ind = np.argpartition(temp_array,-10)[-10:]
    TopTen_ind[i,:] = temp_ind[np.argsort(temp_array[temp_ind])] # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

print(TopTen_ind)

"""
Apply most important pixels to test dataset
"""
ImageTest_full = idx2numpy.convert_from_file(file_ImageTest)
LabelTest_full = idx2numpy.convert_from_file(file_LabelTest)

numPixels = ImageTest_full.shape[1] # images are square and isotropic

ImageTest = np.zeros((10,numPixels,numPixels)) # 10 is the number of basic objects
LabelTest = np.zeros((10,1), dtype=int)

n_LongEnough = 100 # Just need enough iterations to get one of each object

for i in np.arange(n_LongEnough):
    ImageTest[(LabelTest_full[i]+9) % 10,:,:] = ImageTest_full[i,:,:] # must be consistent with ("1","2",..."0") interpretation
    LabelTest[(LabelTest_full[i]+9) % 10] = LabelTest_full[i]

# Transform MNIST labels to our labels, defined in assignment
Label_temp = np.zeros((LabelTest.shape[0],10))
for k in np.arange(LabelTest.shape[0]):
    Label_temp[k,(LabelTest[k]+9) % 10] = 1

LabelTest = Label_temp

ImageTest = ImageTest.reshape((ImageTest.shape[0],numPixels*numPixels))
ImageTest_TopTen = np.zeros((ImageTest.shape[0],numPixels*numPixels)) # reshaped images with just the most important pixels

for k in np.arange(ImageTest_TopTen.shape[0]):
    for l in np.arange(TopTen_ind.shape[1]):
        ImageTest_TopTen[k,TopTen_ind[k,l]] = ImageTest[k,TopTen_ind[k,l]]

## Predict using most important pixels
LabelPrediction_lam = np.zeros((lam.shape[0],sizeLabels,sizeLabels))
for i in np.arange(LabelPrediction_lam.shape[0]):
    LabelPrediction_lam[i,:,:] = np.matmul(ImageTest_TopTen,X_regLasso[i,:,:].T)

print(LabelPrediction_lam)

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

figPredict, axs = plt.subplots(6,2)
for j in np.arange(axs.shape[0]):
    ax = axs[j,0]
    c = ax.pcolor(LabelPrediction_lam[j,:,:]/np.amax(LabelPrediction_lam[j,:,:]))
    figPredict.colorbar(c, ax=ax)
    ax = axs[j,1]
    c = ax.pcolor(LabelTest)

plt.subplots_adjust(hspace = 0.5)

plt.show()
