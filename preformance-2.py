import matplotlib.pyplot as plt
from CGMethod import CGMethod
import numpy as np
# make sure you have installed ssgetpy
import ssgetpy

# to read from file into np.array
from io import StringIO
from scipy.io import mmread


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
    '''
    # Define System of Equations
    A = np.array([[9,3],[3,1]])
    b = np.array([-0.5,1])
    # initial guess
    x0 = np.array([0,0])

    driver(A, b, x0)
    '''
    
    f = open("matrix4.mtx", "r", encoding="utf-8")
    text = f.read()
    m = mmread(StringIO(text))
    A = m.todense()

    b = np.random.rand(14)
    x0 = np.zeros(14)

    driver(A, b, x0)

    
