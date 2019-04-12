import sys
import os
sys.path.append("D:/python/Lib/site-packages")
import numpy as np
import pandas as pd
import scipy.linalg as linalg
import cv2
import operator
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from PIL import Image
import gc

def split70 ():

    for index, row in DF.iterrows():

        if (int(index))%10 > 2 :

            text_train.append(row.text)
            label_train.append(row.label)
        else :
            text_test.append(row.text)
            label_test.append(row.label)
    print(len(text_train),len(label_train))
    print(len(text_test),len(label_test))
    return
def PCA() :
    accs=[]
    alphas=[0.8,0.85,0.9,0.95]
    training_data=np.asarray(text_train)
    training_labels=np.asarray(label_train)
    testing_data=np.asarray(text_test)
    testing_labels=np.asarray(label_test)
    meanVector = np.mean(training_data, 0)
    print(meanVector)

    centeredMatrix = training_data - meanVector
    print("----------------\n Centered Matrix \n----------------")
    print(centeredMatrix)

    print("----------------- \nCovariance Matrix \n-----------------")
    covarianceMatrix = np.dot(centeredMatrix.T, centeredMatrix) * (1 / training_data.shape[0])
    #covarianceMatrix=np.cov(centeredMatrix)
    print(covarianceMatrix)
    print(covarianceMatrix.shape)


    eigenValues ,eigenVector = np.linalg.eigh(covarianceMatrix)
    # eigenValues = np.matrix(np.load('eigenValues.npy'))
    # eigenVector = np.matrix(np.load('eigenVectors.npy'))
    print("--------------- \n eigen values \n --------------")
    print(eigenValues)
    print(eigenValues.shape)
    print("--------------- \n eigen Vectors \n --------------")
    print(eigenVector)
    np.save('eigenValues', eigenValues)
    np.save('eigenVectors', eigenVector)
    eigen_sum = eigenValues.sum()
    for alpha in alphas :
        my_sum = 0
        i=0;
        for x in eigenValues:
            my_sum += x
            i+=1
            if my_sum/eigen_sum >= alpha:
                break
        print(x)
        projection_matrix = eigenVector[:,0:i-1]
        print(projection_matrix.shape)
        projected_data = np.dot(training_data , projection_matrix)
        ks=[1,3,5,7]
        acc=[]
        for k in ks :         
            kneighboursClassifier = KNeighborsClassifier(n_neighbors=k)
            kneighboursClassifier.fit(projected_data, training_labels.T)

            print(kneighboursClassifier.score(np.dot(testing_data,projection_matrix), testing_labels))
            acc.append(kneighboursClassifier.score(np.dot(testing_data,projection_matrix), testing_labels))
        accs.append(acc)        
    return accs

def pca () :

    accs=[]

    alpha=[0.8,0.85,0.9,0.95]
    accs=PCA()
    print(accs)
    tst=(np.asarray(accs))[:,0]
    print(tst)
    plt.plot(alpha,tst)
    plt.show()
    i=0
    mean=[]

    while i<4:
        mean.append(np.mean((np.asarray(accs))[:,i]))
        i+=1
    ks=[1,3,5,7]    
    plt.plot(ks,mean)
    plt.show()
    return 
def LDA() :
    samples=7
    class_matrices = list()
    class_means = list()
    Si= list()
    x = 0
    for i in range(0, 40):
        class_matrices.append(text_train[x:x+samples])
        class_means.append(np.mean(class_matrices[i], axis=0))
        x += samples
    total=0  
    #print(len(class_matrices))
    #print(len(class_means))
    #total_mean = np.mean(text_train, axis=0)
    for i in range(0,40):
        total+=(7/40)*class_means[i]    
    Sb = np.zeros((10304, 10304))
    print("calculating Sb ...")
    for i in range(0, 40):
        saw = class_means[i] - total
        Sb += samples * (np.dot(saw ,saw.T))
    print("calculated Sb")
    print(Sb)
    
    #calculating Si
    print("calculating Si")
    for i in range(0, 40):
        class_matrices[i] = class_matrices[i] - class_means[i]  #centering data (Z)
        
    for i in range(0, 40):
            Si.append(np.dot(class_matrices[i].ravel().T,class_matrices[i].ravel()))
    print("calculated Si")
    print(Si)
    S = np.zeros((10304,10304))
    for i in range(0, 40):
        S =S + (Si[i])
        
    print("doneeeee")
    print(S)
    Sinv = np.linalg.pinv(S)
    print("Inverse calculated")
    print(Sinv)
    np.savetxt('inversemat.txt',Sinv,fmt='%f')
    
    eigens = np.linalg.eigh(Sinv * Sb)
    print("calculated eigen values/vectors")
    eigen_values = eigens[0]
    eigen_vectors = eigens[1]
    print("values")
    print(eigen_values)
    print("vectors")
    print(eigen_vectors)
    projectionMatrix = eigen_vectors[:, eigen_vectors.shape[1]-39:eigen_vectors.shape[1]]
    projected_data = np.dot(text_train,projectionMatrix)
    ks=[1,3,5,7,9,11,13,15,17]
    accs=[]
    for k in ks :    
        kneighboursClassifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        kneighboursClassifier.fit(projected_data, (np.asmatrix(label_train).ravel()).T)
        score=kneighboursClassifier.score(np.dot(text_test,projectionMatrix), np.asmatrix(label_test).ravel().T)
        accs.append(score)
        print(score)
    plt.plot(ks,accs)
    plt.show()
    return


A = []  # A will store list of image vectors
Label = [] # Label will store list of identity label
directory= "C:/Users/DELL/Documents/Pattern Recognition/ass1/att_faces/orl_faces"
# browsing the directory
for f in os.listdir(directory):
    if f[-3:] =='file':
        continue
    infile = os.path.join(directory, f)
    for f2 in os.listdir(infile):
        print(f2)
        #im = cv2.imread(f2,0)
        im= Image.open(os.path.join(infile, f2))
        
        im_vec = np.reshape(im, -1)
       
        A.append(im_vec)
        Label.append(f[1:])
        

faces = np.array(A, dtype=np.float32)
faces = faces.T
#idLabel = np.array(Label)
#print(Label)
#print(len(Label))
#print(len(A))
#print(A)
# create a dataframe using texts and lables
DF = pd.DataFrame()
df_test=pd.DataFrame()
df_train=pd.DataFrame()
DF['label'] = Label
DF['text'] = A
#print(DF)
text_train=[]
text_test=[]
label_train=[]
label_test=[]
for index, row in DF.iterrows():
   # print(index)
   # print(row)
    if (int(index))%2==0 :
        
        text_train.append(row.text)
        label_train.append(row.label)
    else :
        text_test.append(row.text)
        label_test.append(row.label)


text_train=[]
text_test=[]
label_train=[]
label_test=[]
split70()
print(len(text_test))
print(len(text_train))
#print(label_test)
#print(label_train)


pca ()
LDA()