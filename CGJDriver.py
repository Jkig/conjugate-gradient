import matplotlib.pyplot as plt
from CGMethod import CGMethod
from jacobihybrid import CGJMethod
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
    x,xLst,its,ier = CGJMethod(A,b,x0,Nmax,tol)
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

    A = np.array([[40000,2,1,1,5],[2,-1,1,1,3],[1,1,1,2,2],[1,1,2,-1,1],[5,3,2,1,9]])
    b = np.array([-0.5, 1, 1,4, 6])
    x0 = np.array([0,0,0,0,0])

    driver(A,b,x0)


    A = np.array([[400000000,2,1,1],[2,-1,1,1],[1,1,1,2],[1,1,2,-1]])
    b = np.array([-0.5, 1, 1,4])
    x0 = np.array([0,0,0,0])

    driver(A,b,x0)

    A=np.array([[ 1.00665013, -1.41434036, -0.4953264,  -0.43520063],
                [-1.41434036, 3.81637179,  0.58656485,  0.5053143 ],
                [-0.4953264,   0.58656485,  0.87865608,  0.00746661],
                [-0.43520063,  0.5053143,   0.00746661,  0.79932913]] )
    b=np.array([-0.99475612,  2.58211651, 0.68135327,  0.56259304])
    x0 = np.array([0,0,0,0])
    driver(A,b,x0)


    f = open("pref.2.matrix4.mtx", "r", encoding="utf-8")
    text = f.read()
    m = mmread(StringIO(text))
    A = m.todense()
    A = np.array(A)

    b = np.random.rand(14)
    x0 = np.zeros(14)
    driver(A,b,x0)
