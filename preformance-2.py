import matplotlib.pyplot as plt
from CGMethod import CGMethod
import numpy as np
# make sure you have installed ssgetpy
import ssgetpy

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
    # Define System of Equations
    A = np.array([[9,3],[3,1]])
    b = np.array([-0.5,1])
    # initial guess
    x0 = np.array([0,0])
    
    # here we are trying a basic 2x2 matrix with det = 0
    # driver(A,b,x0)

    # as we see, x* goes to nan,nan, and maxes out iterations, in other words,
    #   it does not converge
    #   it av goes to zero, then <v, av> is zero, then we have div by zero and
    #   nan shows up
    
    
    # did mandy write this,, is dot product right? and does it matter?
    
    # i'm going to pull a bigger mostly positive definite matrix with det=0
    # i wanna do more testing here.
    
    n = (10, 12)
    a = ssgetpy.search(rowbounds = n, colbounds = n, limit = 1)[0]
    a.download(destpath = '~/derek/conjugate-gradient')
    print(a)
    
