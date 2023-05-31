
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 


train_path = str(sys.argv[1])
test_path = str(sys.argv[2])

dfx = pd.read_csv(train_path+"/"+"X.csv",header=None)
dfy = pd.read_csv(train_path+"/"+"Y.csv",header=None)



testx = pd.read_csv(test_path+"/"+"X.csv",header=None)
testx = testx.to_numpy()

x=dfx.to_numpy()
y_c=dfy.to_numpy()
y=np.zeros((len(y_c),1))
#Alaska = 0 Canada =1 
def Normalize(x):
    x_m = np.mean(x,axis=0)
    # print(x_m)
    x_v = np.std(x,axis=0)
    #print(x_v)
    x=(x-x_m)/x_v 
    return x
x=Normalize(x)
testx = Normalize(testx)
x1 = []
x2 = []
m1 = np.array([[0,0]])
m2 = np.array([[0,0]])
c_alas=0
for i in range(len(y_c)):
    if(y_c[i,0]=="Alaska"):
        y[i,0]=0
        x1.append(x[i])
        c_alas=c_alas+1
        m1=m1+x[i]
    else:
        y[i,0]=1
        x2.append(x[i])
        m2 = m2+x[i]
m1 = m1/c_alas
m2 = m2/(len(x)-c_alas)
# print(m1)
# print(m2)
E1 = np.zeros((2,2))
for i in range(len(x1)):
    E1 = E1+np.dot((x1[i]-m1).transpose(),(x1[i]-m1))
E2 = np.zeros((2,2))
for i in range(len(x2)):
    E2 = E2+np.dot((x2[i]-m2).transpose(),(x2[i]-m2))
E = np.zeros((2,2))
E=E+E1+E2
E1=E1/c_alas
E2=E2/(len(x)-c_alas)
E=E/len(x)
x1=np.array(x1)
x2=np.array(x2)
pi = c_alas/len(y)

def show_lin():
    xx = np.linspace(-3, 3, 100)
    yy = np.linspace(-3, 3, 100)
    xm, ym = np.meshgrid(xx, yy)
    z=np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            xs=np.zeros((1,2))
            xs[:,0]=xx[i]
            xs[:,1]=yy[j]
            d1 = np.linalg.det(E) 
            x_m1 = xs-m1
            x_m2 = xs-m2
            z[i,j]=np.dot(x_m2,np.dot(np.linalg.inv(E),x_m2.transpose()))-np.dot(x_m1,np.dot(np.linalg.inv(E),x_m1.transpose()))+2*np.log((1-pi)*(math.sqrt(d1))/((pi)*math.sqrt(d1)))
    plt.scatter(x1[:,0],x1[:,1],color='blue')
    plt.scatter(x2[:,0],x2[:,1],color='yellow')
    plt.contour(xm,ym,z,levels=[0],color='red')
    plt.show()
    return

def show_quad():
    xx = np.linspace(-3, 3, 100)
    yy = np.linspace(-3, 3, 100)
    xm, ym = np.meshgrid(xx, yy)
    z=np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            xs=np.zeros((1,2))
            xs[0,0]=xm[i][j]
            xs[0,1]=ym[i][j]
            d1 = np.linalg.det(E1) 
            d2 = np.linalg.det(E2)
            x_m1 = xs-m1
            x_m2 = xs-m2
            z[i,j]=np.dot(x_m2,np.dot(np.linalg.inv(E2),x_m2.transpose()))-np.dot(x_m1,np.dot(np.linalg.inv(E1),x_m1.transpose()))+2*np.log((1-pi)*(math.sqrt(d2))/((pi)*math.sqrt(d1)))
    plt.scatter(x1[:,0],x1[:,1],color='blue')
    plt.scatter(x2[:,0],x2[:,1],color='yellow')
    plt.contour(xm,ym,z,levels=[0],color='orange')
    plt.show()
    return 

#show_quad()

def predict():
    n = len(testx)
    nf = open("result_4.txt","w")
    for i in range(n):
        xs=np.zeros((1,2))
        xs[0,0]=testx[i,0]
        xs[0,1]=testx[i,1]
        x_m1 = xs-m1
        x_m2 = xs-m2
        d1 = np.linalg.det(E1) 
        d2 = np.linalg.det(E2)
        if(np.dot(x_m2,np.dot(np.linalg.inv(E2),x_m2.transpose()))-np.dot(x_m1,np.dot(np.linalg.inv(E1),x_m1.transpose()))+2*np.log((1-pi)*(math.sqrt(d2))/((pi)*math.sqrt(d1)))>0):
            nf.write("Alaska\n")
        else:
            nf.write("Canada\n")
    nf.close()
    return
predict()





