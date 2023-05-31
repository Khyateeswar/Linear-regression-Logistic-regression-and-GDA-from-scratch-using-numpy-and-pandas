from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import sys

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

x_in = pd.read_csv(train_path+"/"+'X.csv',header=None)
y_in = pd.read_csv(train_path+"/"+'Y.csv',header=None)

testx = pd.read_csv(test_path+"/"+"X.csv",header=None)
testx = testx.to_numpy()


x=x_in.to_numpy()
y=y_in.to_numpy()

def Normalize(x):
    x_m = np.mean(x,axis=0)
    x_v = np.std(x,axis=0)
    for i in range(len(x)):
        x[i,0]=(x[i,0]-x_m)/x_v
    return x
    
x_prev = x
y_prev = y
x = Normalize(x)
testx = Normalize(testx)

x=np.insert(x,0,1,axis=1)
testx = np.insert(testx,0,1,axis=1)
#print(x)
theta = np.array([[0],[0]])
eta=0.01
epsilon=0.000001
s_theta = []

def Grad_Desc(f_x,f_theta,f_y):
    global s_theta
    f_eta=0.025
    h=np.dot(f_x,f_theta)
    er = y-h
    #print(er)
    j=((np.dot(er.transpose(),er))[0,0])/(2*len(f_x))
    j1=10
    i=0
    while(abs(j-j1)>0.000000000001):
        #time.sleep(0.2)
        #plt.scatter(x_prev,f_y)
        #plt.plot(x_prev,h,color='red')
        #plt.show(block=False)
        s_theta.append([f_theta[0,0],f_theta[1,0]])
        f_theta = f_theta+f_eta*(1/len(f_x))*(np.dot(f_x.transpose(),er))
        h=np.matmul(f_x,f_theta)
        er = y-h
        j1=j
        j=((np.dot(er.transpose(),er))[0,0])/(2*len(f_x))
        #print(f_theta[0,0])
        #print(f_theta[1,0])
        #print(" ")
        """
        if(i%100==0):
            plt.clf()
            plt.scatter(x_prev,f_y,color='blue')
            plt.plot(x_prev,h,color='orange')
            plt.show(block=False)
            plt.pause(0.2)
        """
        i=i+1
    #print(i)
    #print(f_theta[0,0])
    #print(f_theta[1,0])
    """
    plt.scatter(f_x[:,1],f_y,color='blue')
    plt.plot(f_x[:,1],h,color='red')
    plt.show(block=False)
    plt.pause(4)
    plt.clf()
    """
    return i

count = Grad_Desc(x,theta,y)
s_theta = np.array(s_theta)

def show_co(s_theta,x,y):
    n = len(s_theta)
    cx = np.linspace(-1, 2.5, 50)
    cy = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(cx, cy)
    z = np.zeros((50,50))
    for i in range(50):    
        for j in range(50):
            xs=np.zeros((2,1))
            xs[0,0]=X[i][j]
            xs[1,0]=Y[i][j]
            h=np.matmul(x,xs)
            er=y-h
            z[i,j]=  ((np.dot(er.transpose(),er))[0,0])/(2*len(x))
            #print(z[i,j])
    plt.contour(X,Y,z,levels=[0.25,0.5,1,2],color='blue')
    plt.show(block=False)
    plt.pause(0.2)
    for i in range(n):
        if(i%1==0):
            plt.scatter(s_theta[i,0],s_theta[i,1],color='red')
            plt.show(block=False)
            plt.pause(0.002)
    return 

def show_3d(s_theta,x,y):
    n = len(s_theta)
    cx = np.linspace(-1, 2.5, 50)
    cy = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(cx, cy)
    z = np.zeros((50,50))
    for i in range(50):    
        for j in range(50):
            xs=np.zeros((2,1))
            xs[0,0]=X[i][j]
            xs[1,0]=Y[i][j]
            h=np.matmul(x,xs)
            er=y-h
            z[i,j]=  ((np.dot(er.transpose(),er))[0,0])/(2*len(x))
            #print(z[i,j])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X,Y, z, 50, cmap=cm.coolwarm)
    ax.view_init(45,40)
    for i in range(n):
        if(i%100==0):
            sx=np.zeros((2,1))
            sx[0,0]=s_theta[i,0]
            sx[1,0]=s_theta[i,1]
            h=np.matmul(x,sx)
            er=y-h
            zs=  ((np.dot(er.transpose(),er))[0,0])/(2*len(x))
            ax.scatter(s_theta[i,0],s_theta[i,1],zs,color='red')
            plt.show(block=False)
            plt.pause(0.2)
    return 

#show_co(s_theta,x,y)

def predict():
    theta = np.zeros((2,1))
    theta[0,0]= s_theta[len(s_theta)-1,0]
    theta[1,0]= s_theta[len(s_theta)-1,1]
    pred_y=np.matmul(testx,theta)
    nf = open("result_1.txt","w")
    #print(pred_y)
    for a in pred_y:
        nf.write(str(a[0]))
        nf.write("\n")
    nf.close()
    return
predict()



