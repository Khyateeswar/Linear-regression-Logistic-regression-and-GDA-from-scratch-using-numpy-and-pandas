from distutils.log import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

test_path = str(sys.argv[1])

testx = pd.read_csv(test_path+"/"+'X.csv',header=None)
testx = testx.to_numpy()


x1 = np.random.normal(3,2,1000000)
x2 = np.random.normal(-1,2,1000000)
eps = np.random.normal(0,math.sqrt(2),size=(1000000,1))
x=np.zeros((len(x1),2))
x[:,0] = x1
x[:,1] = x2
g_theta = [[3],[1],[2]]
x_prev=x
x=np.insert(x,0,1,axis=1)
testx = np.insert(testx,0,1,axis=1)
y=np.dot(x,g_theta)+eps
f_theta =  [[0],[0],[0]]
m=100
s_theta=[]



def Grad_Desc(f_x,f_y,start,length):
    global f_theta
    f_eta=0.001
    f_x = f_x[start:start+length]
    f_y = f_y[start:start+length]
    h=np.dot(f_x,f_theta)
    er = f_y-h
    j=((np.dot(er.transpose(),er))[0,0])/(2*len(f_x))
    f_theta = f_theta+f_eta*(1/len(f_x))*(np.dot(f_x.transpose(),er))
    ans = np.array([[f_theta[0,0]],[f_theta[1,0]],[f_theta[2,0]]])
    """
    print(f_theta[0,0])
    print(f_theta[1,0])
    print(f_theta[2,0])
    print("    ")
    """
    return ans
c=0
er=0.1
er1=0
while(abs(er-er1)>0.000000001):
    i=0
    while(i+m<len(x)):
        t=Grad_Desc(x,y,i,m)
        s_theta.append([t[0,0],t[1,0],t[2,0]])
        i = i+m
    er1=er
    t=Grad_Desc(x,y,i,len(x)-i)
    s_theta.append([t[0,0],t[1,0],t[2,0]])
    er=+((np.dot((y-np.matmul(x,t)).transpose(),(np.matmul(x,t))))[0,0])/(2*len(x))
    c=c+1
#print(c)
s_theta=np.array(s_theta)

def show_an3d(s_theta):
    n = len(s_theta)
    print(n)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.view_init(45,40)
    for i in range(n):
        if(i%1==0):
            ax.scatter(s_theta[i,0],s_theta[i,1],s_theta[i,2],color='red')
            plt.show(block=False)
            plt.pause(0.00000002)
    return 

def show_3d(s_theta):
    print(len(s_theta))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(s_theta[:,0],s_theta[:,1],s_theta[:,2],color='red')
    plt.show()

#show_3d(s_theta)
def predict():
    theta = np.zeros((3,1))
    theta[0,0]= s_theta[len(s_theta)-1,0]
    theta[1,0]= s_theta[len(s_theta)-1,1]
    theta[2,0]= s_theta[len(s_theta)-1,2]
    pred_y=np.matmul(testx,theta)
    nf = open("result_2.txt","w")
    #print(pred_y)
    for a in pred_y:
        nf.write(str(a[0]))
        nf.write("\n")
    nf.close()
    return
predict()




