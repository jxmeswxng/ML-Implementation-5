import sys
import numpy as np
from PIL import Image

K = int(sys.argv[1])
inputImage = sys.argv[2]
outputImage = sys.argv[3]

# Prepare dataset
img = Image.open(inputImage)
imgMap = img.load()

pixelData = []
for i in range(img.size[0]):
    for j in range(img.size[1]):
        pixelData.append((imgMap[i,j][0],imgMap[i,j][1],imgMap[i,j][2],i,j))
pixelData = np.array(pixelData).astype(float)

# standardize values of each feature
meanstd = []
for col in range(pixelData.shape[1]):
    mean = np.mean(pixelData[:,col])
    std = np.std(pixelData[:,col])
    for row in range(pixelData.shape[0]):
        pixelData[row,col] = (pixelData[row,col] - mean) / std
    meanstd.append((mean,std))
meanstd = np.asarray(meanstd)

# write Objective Function  
def objectiveFunc(instance,center):
    return sum((instance - center)**2)    

# write K-Means functions
def kMeans(k,X):
    n = 100    
    newCentroids = X[np.random.choice(X.shape[0],k,replace=False)]
    for i in range(n):      
        centroids = newCentroids  
        distList = [] 
        for i in range(X.shape[0]):
            instDistList = []
            for j in range(k):
                instDistList.append(objectiveFunc(X[i,:],centroids[j]))
            distList.append(instDistList)
        minDistList = []
        for i in range(X.shape[0]):
            minDistList.append(np.argmin(distList[i]))
        pixelDataLabels = np.hstack((np.asarray(minDistList).reshape((np.asarray(minDistList).shape[0],1)),X))
        meanList = []
        for i in range(k):
            meanList.append(np.mean(pixelDataLabels[pixelDataLabels[:,0]==i],axis=0))
        meanArray = np.zeros((1,pixelDataLabels.shape[1]))
        for i in meanList:
            meanArray = np.vstack((meanArray,np.array(i)))
        newCentroids = np.delete(np.delete(meanArray,(0),axis=0),(0),axis=1)
        if sum(sum((centroids-newCentroids)**2)) < 0.0000001:
            dataList = []
            for i in range(k):
                dataList.append(pixelDataLabels[pixelDataLabels[:,0]==i])
            return (dataList,meanList)
        dataList = []
        for i in range(k):
            dataList.append(pixelDataLabels[pixelDataLabels[:,0]==i])
        return (dataList,meanList)

# run K-Means
dataCentroids, centroids = kMeans(K,pixelData)

# generate pixel colors
centroidsList = []
for i in range(K):
    centroid = np.array(centroids[i])
    centroid = centroid.reshape(1,6)
    centroid = centroid[:,1:4]
    centroidsList.append(centroid)

dataCentroidsList = []
for i in range(K):
    dataCentroidsList.append(np.array(dataCentroids[i])[:,1:6])

for i in range(K):
    dataCentroidsList[i][:,0:3] = centroidsList[i]

newPixelData = np.zeros((1,5))
for i in range(K):
    newPixelData = np.vstack((newPixelData,dataCentroidsList[i]))
newPixelData = newPixelData[1:(img.size[0] * img.size[1] + 1),:]

# inverse standardization
for col in range(newPixelData.shape[1]):
    mean = meanstd[col][0]
    std = meanstd[col][1]
    for row in range(newPixelData.shape[0]):
        newPixelData[row,col] = np.int(np.around((newPixelData[row,col] * std) + mean))

# sort & clean pixels
sortedPixelList = sorted(newPixelData,key=lambda x : (x[3],x[4]))
sortedPixelData = np.asarray(sortedPixelList)
feedPixelData = sortedPixelData[:,0:3]
feedPixelData = feedPixelData.astype(int)
feedPixelList = [tuple(i) for i in feedPixelData]

# create new image
newImg = Image.new('RGB',(img.size[0],img.size[1]),"black")
newImgMap = newImg.load()

count = 0
for i in range(newImg.size[0]):
    for j in range(newImg.size[1]):
        newImgMap[i,j] = feedPixelList[count]
        count = count + 1

# output new image
newImg.save(outputImage)