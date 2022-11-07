import matplotlib.pyplot as plt
from CGMethod import CGMethod
import numpy as np

def driver():
    #Define System of Equations
    A = np.array([[9,0],[0,1]])
    b = np.array([-0.5,1])
    #Set exit parameters
    Nmax = 100
    tol = 1.0e-8
    #initial condition
    #how to use other initial conditions (how do we define v_0)
    x0 = np.array([0,0])
    #run evaluator
    x,xLst,its,ier = CGMethod(A,b,x0,Nmax,tol)
    print("x = ", x)
    print("Number of Iterations: ", its)

    '''plotting'''
    #could someone else work out plotting?





if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()
