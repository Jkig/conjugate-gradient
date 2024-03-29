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
    tol = .1 #1.0e-8
    #initial condition
    #run evaluator
    x,xLst,its,ier = CGMethod(A,b,x0,Nmax,tol)
    # print("x = ", x)
    # print("Number of Iterations: ", its)

    '''plotting'''
    # list of abs errors vs last one
    abs_err = []
    last_index = len(xLst) - 1
    
    for i in range(len(xLst)):
        error = np.linalg.norm(xLst[i]-xLst[last_index])
        abs_err.append(error)

    # this is my edit in the driver:
    return its
    
    plt.plot(range(0,len(xLst)),abs_err)
    plt.show()



if __name__ == '__main__':
    
    # test case
    '''
    xlist_past = []
    for i in range(40):
        A = np.array([[9,0,1],[0,1,2],[1,2,1]])
        b = np.array([-0.5, 1, 1])
        x0 = np.array([0,0,0])
        xlist_past.append(driver(A, b, x0))
    print(sum(xlist_past)/len(xlist_past))

    print(np.linalg.det(A))

    xlist_past = []
    for i in range(40):
        A = np.array([[9,0,1],[0,1,2],[1,2,1]])
        b = np.array([-0.5, 1, 1])
        x0 = np.array([0,0,0])

        A[0,2] = -2
        
        xlist_past.append(driver(A, b, x0))
    print(sum(xlist_past)/len(xlist_past))

    print(np.linalg.det(A))
    '''
    
    '''
    # case 1 - not fully positive definite (first sub determinets = 0)
    A = np.array([[5,2,1,1],[2,-1,1,1],[1,1,1,2],[1,1,2,-1]])
    b = np.array([-0.5, 1, 1,4])
    x0 = np.array([0,0,0,0])
    print("case 1")
    driver(A,b,x0)

    '''
    '''
    # case 2 - ill conditioned number 1, k = 5
    A = np.array([[40000,2,1,1,5],[2,-1,1,1,3],[1,1,1,2,2],[1,1,2,-1,1],[5,3,2,1,9]])
    b = np.array([-0.5, 1, 1,4, 6])
    x0 = np.array([0,0,0,0,0])
    print()
    print("case 2:", np.linalg.cond(A), "results:")
    driver(A,b,x0)
    '''
    
    
    '''
    # case 3 - ill conditioned number 2, k = 10
    A = np.array([[400000000,2,1,1],[2,-1,1,1],[1,1,1,2],[1,1,2,-1]])
    b = np.array([-0.5, 1, 1,4])
    x0 = np.array([0,0,0,0])
    print()
    print("case 3:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A,b,x0)


    # case 4 - good example of poorly conditioned matrix
    f = open("pref.2.matrix4.mtx", "r", encoding="utf-8")
    text = f.read()
    m = mmread(StringIO(text))
    A = m.todense()
    A = np.array(A)

    b = np.random.rand(14)
    x0 = np.zeros(14)
    
    print()
    print("case 4, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A, b, x0)
    
    
    # case 5 - not perfect, but converges in 30, 31, or 32 depending on the trial
    f = open("pref.2.matrix5.mtx", "r", encoding="utf-8")
    text = f.read()
    m = mmread(StringIO(text))
    A = m.todense()
    A = np.array(A)

    b = np.random.rand(24)
    x0 = np.zeros(24)
    
    print()
    print("case 5, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A, b, x0)
    '''
    
    
    # case 5
    xlist_past = []
    for i in range(40):
        f = open("pref.2.matrix5.mtx", "r", encoding="utf-8")
        text = f.read()
        m = mmread(StringIO(text))
        A = m.todense()
        A = np.array(A)

        b = np.random.rand(24)
        x0 = np.zeros(24)
    
        # print()
        # print("case 5, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))

        xlist_past.append(driver(A, b, x0))
    print(sum(xlist_past)/len(xlist_past))


    # case 5
    xlist_past = []
    for i in range(40):
        f = open("pref.2.matrix5.mtx", "r", encoding="utf-8")
        text = f.read()
        m = mmread(StringIO(text))
        A = m.todense()
        A = np.array(A)

        b = np.random.rand(24)
        x0 = np.zeros(24)
    
        # print()
        # print("case 5, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))


        A[2,1] = .7
        A[1,2] = .8
        
        xlist_past.append(driver(A, b, x0))
    print(sum(xlist_past)/len(xlist_past))

    print(np.linalg.det(A))
    
