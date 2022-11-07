import matplotlib.pyplot as plt
from CGMethod import CGMethod
import numpy as np

def driver(A,b,x0):
    #Set exit parameters
    Nmax = 100
    tol = 1.0e-8
    #initial condition
    #run evaluator
    x,xLst,its,ier = CGMethod(A,b,x0,Nmax,tol)
    print("x = ", x)
    print("Number of Iterations: ", its)

    '''plotting'''
    # list of abs errors vs last one
    abs_err = []
    last_index = len(xLst) - 1
    
    for i in range(len(xLst)):
        error = np.linalg.norm(xLst[i]-xLst[last_index])
        abs_err.append(error)
    
    plt.plot(range(0,len(xLst)),abs_err)
    plt.show()



if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    #Define System of Equations
    A = np.array([[9,0],[0,1]])
    b = np.array([-0.5,1])
    #how to use other initial conditions (how do we define v_0)
    x0 = np.array([0,0])
    
    driver(A,b,x0)
    
    A = np.array([[9,0,1],[0,1,2],[1,2,1]])
    b = np.array([-0.5, 1, 1])
    x0 = np.array([0,0,0])
    driver(A,b,x0)

    A = np.array([[1,-1,1,0],[-1,4,2,1],[1,2,12,2],[0,1,2,6]])
    b = np.array([-0.5, 1, 1,4])
    x0 = np.array([0,0,0,0])
    driver(A,b,x0)
