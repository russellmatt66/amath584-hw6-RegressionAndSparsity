"""
Matt Russell
AMATH584 hw6-M2
12/15/20
"""
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

file_ImageTrain = 'train-images-idx3-ubyte'
file_LabelTrain = 'train-labels-idx1-ubyte'

ImageTrain_full = idx2numpy.convert_from_file(file_ImageTrain)
LabelTrain_full = idx2numpy.convert_from_file(file_LabelTrain)

numPixels = ImageTrain_full.shape[1] # images are square and isotropic

ImageTrain = np.zeros((10,numPixels,numPixels)) # 10 is the number of basic objects, ndarray to contain one instance of each basic object
LabelTrain = np.zeros((10,1), dtype=int)

n_LongEnough = 100 # Just need enough iterations to get one of each object

for i in np.arange(n_LongEnough):
    ImageTrain[(LabelTrain_full[i]+9) % 10,:,:] = ImageTrain_full[i,:,:] # must be consistent with ("1","2",..."0") interpretation
    LabelTrain[(LabelTrain_full[i]+9) % 10] = LabelTrain_full[i]

Label_temp = np.zeros((LabelTrain.shape[0],10)) # 10 is the dimension of space the assignment labels sit in
for k in np.arange(LabelTrain.shape[0]):
    Label_temp[k,(LabelTrain[k]+9) % 10] = 1

LabelTrain = Label_temp

ImageTrain = ImageTrain.reshape((ImageTrain.shape[0],numPixels*numPixels))

## LASSO for several different regularizations
lam = np.array([0.1,0.25,0.5,0.75,1.0,2.0])
X_regLasso = np.zeros((lam.shape[0],LabelTrain.shape[1],ImageTrain.shape[1])) #(numRegularizations,sizeLabels,numPixels^2), interpretation of axis = 1 is ("1","2",...,"0")

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
MIP_ind = np.zeros((LabelTrain.shape[1],n_important),dtype=int) # axis = 0 represents each of the different labels with the same interpretation of ("1","2",..."0"), the indices of the pixels with the largest weightings for the respective label are stored in ascending order along axis = 1
for i in np.arange(X_Average.shape[0]):
    temp_array = X_Average[i,:]
    temp_ind = np.argpartition(temp_array,-10)[-10:]
    MIP_ind[i,:] = temp_ind[np.argsort(temp_array[temp_ind])] # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

print(MIP_ind)

MIP_Visualizer = np.zeros((LabelTrain.shape[0],numPixels*numPixels))

for i in np.arange(MIP_ind.shape[0]):
    for j in np.arange(MIP_ind.shape[1]):
        MIP_Visualizer[i,MIP_ind[i,j]] = 1

"""
Plotting
"""
plt.pcolor(MIP_Visualizer)
plt.title('Most Important Pixels for each Digit')
plt.xlabel('Pixel')
plt.ylabel('Particular Label (0,1,...8,9) = ("1","2",..."9","0")')

plt.show()
