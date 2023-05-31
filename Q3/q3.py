import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

x_in = pd.read_csv(train_path+"/"+'X.csv',header=None)
y_in = pd.read_csv(train_path+"/"+'Y.csv',header=None)

testx = pd.read_csv(test_path+"/"+"X.csv",header=None)
testx = testx.to_numpy()

x=x_in.to_numpy()
y=y_in.to_numpy()

x=np.insert(x,0,1,axis=1)
testx = np.insert(testx,0,1,axis=1)
theta = np.array([[0],[0],[0]])

def new_ral(x,theta,y):
    h = np.matmul(x,theta)
    sig = 1/(1+np.exp(-h))
    g=np.dot(x.transpose(),sig-y)
    dia = np.zeros((len(x),len(x)))
    j=np.dot(y.transpose(),np.log(sig))+np.dot((1-y).transpose(),np.log(1-sig))
    j1=0
    for i in range(len(x)):
        dia[i,i]=sig[i,0]*(1-sig[i,0])
    hes = np.dot(x.transpose(),np.dot(dia,x))
    while(abs(j-j1)>0.00000001):
        theta = theta - np.dot(np.linalg.inv(hes),g)
        h = np.matmul(x,theta)
        sig = 1/(1+np.exp(-h))
        g=np.dot(x.transpose(),sig-y)
        dia = np.zeros((len(x),len(x)))
        for i in range(len(x)):
            dia[i,i]=sig[i,0]*(1-sig[i,0])
        hes = np.dot(x.transpose(),np.dot(dia,x))
        j1=j
        j=np.dot(y.transpose(),np.log(sig))+np.dot((1-y).transpose(),np.log(1-sig))
        #print(j)
    #print("converged")
    # print(theta[0,0])
    # print(theta[1,0])
    # print(theta[2,0])
    xmin = np.amin(x[:,1])
    ymin = -(theta[0,0]*1 + theta[1,0]*xmin)/theta[2,0]
    xmax = np.amax(x[:,1])
    ymax = -(theta[0,0]*1 + theta[1,0]*xmax)/theta[2,0]
    x_a = np.array([xmin,xmax])
    y_a = np.array([ymin,ymax])
    x1 = []
    x2 = []
    for i in range(len(y)):
        if(y[i,0]==0):
            x1.append(x[i])
        else:
            x2.append(x[i])
    x1=np.array(x1)
    x2=np.array(x2) 
    # plt.scatter(x1[:,1],x1[:,2],color='blue')
    # plt.scatter(x2[:,1],x2[:,2],color='yellow')
    # plt.plot(x_a,y_a,color='red')
    # plt.show()
    return theta

final_t = new_ral(x,theta,y)

def predict():
    pred_y=np.matmul(testx,final_t)
    nf = open("result_3.txt","w")
    # print(pred_y)
    for a in pred_y:
        if(a[0]>0):
            nf.write("1\n")
        else:
            nf.write("0\n")
    nf.close()
    return

predict()
    



