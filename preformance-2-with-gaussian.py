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

    real_sol = np.linalg.solve(A,b)
    
    for i in range(len(xLst)):
        # error = np.linalg.norm(xLst[i]-xLst[last_index])
        error = np.linalg.norm(xLst[i]-real_sol)
        abs_err.append(error)
    
    plt.plot(range(0,len(xLst)),abs_err)
    plt.show()



if __name__ == '__main__':
    # test case
    A = np.array([[9,0,1],[0,1,2],[1,2,1]])
    b = np.array([-0.5, 1, 1])
    x0 = np.array([0,0,0])
    driver(A,b,x0)
    
    
    # case 7 - ill conditioned, doesn't converge, even in 399 iterations
    f = open("pref.2.matrix7.mtx", "r", encoding="utf-8")
    text = f.read()
    m = mmread(StringIO(text))
    A = m.todense()
    A = np.array(A)

    b = np.random.rand(30)
    x0 = np.zeros(30)
    
    print()
    print("case 7, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A, b, x0)
    print(np.linalg.det(A))

