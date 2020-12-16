"""
Matt Russell
testing file for hw6-M2 project
"""
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

file_ImageTrain = 'train-images-idx3-ubyte'
file_LabelTrain = 'train-labels-idx1-ubyte'
file_ImageTest = 't10k-images-idx3-ubyte'
file_LabelTest = 't10k-labels-idx1-ubyte'

"""
Grab one of each object from test set in order to apply pixels determined to be
most important to it
"""
ImageTest_full = idx2numpy.convert_from_file(file_ImageTest)
LabelTest_full = idx2numpy.convert_from_file(file_LabelTest)

numPixels = ImageTest_full.shape[1] # images are square and isotropic

ImageTest = np.empty((10,numPixels,numPixels)) # 10 is the number of basic objects
LabelTest = np.empty((10,1), dtype=int)

n_LongEnough = 100 # Just need enough iterations to get one of each object

for i in np.arange(n_LongEnough):
    ImageTest[(LabelTest_full[i]+9) % 10,:,:] = ImageTest_full[i,:,:]
    LabelTest[(LabelTest_full[i]+9) % 10] = LabelTest_full[i]

# Transform MNIST labels to our labels, defined in assignment
Label_temp = np.zeros((LabelTest.shape[0],10))
for k in np.arange(LabelTest.shape[0]):
    Label_temp[k,(LabelTest[k]+9) % 10] = 1

LabelTest = Label_temp

print(ImageTest)
print(LabelTest)

ImageTest = ImageTest.reshape((ImageTest.shape[0],numPixels*numPixels))
ImageTest_TopTen = np.zeros((ImageTest.shape[0],numPixels*numPixels))

n_important = 10
sizeLabels = 10
TopTen_ind = np.zeros((sizeLabels,n_important),dtype=int)
for i in np.arange(TopTen_ind.shape[0]):
    for j in np.arange(TopTen_ind.shape[1]):
        TopTen_ind[i,j] = np.random.randint(0,numPixels*numPixels-1)

print(TopTen_ind)

for k in np.arange(ImageTest_TopTen.shape[0]):
    for l in np.arange(TopTen_ind.shape[1]):
        ImageTest_TopTen[k,TopTen_ind[k,l]] = ImageTest[k,TopTen_ind[k,l]]

"""
Plotting

figTopTen, axs = plt.subplots(1,10)
for j in np.arange(axs.shape[0]):
    ax = axs[j]
    c = ax.pcolor(ImageTest_TopTen[j,:,:])

plt.show()
"""
