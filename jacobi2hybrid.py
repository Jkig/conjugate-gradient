from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
import numpy as np

def CGJ2Method(A,b,x0,Nmax,tol):
    D = diag(A)
    R = A - diagflat(D)
    x0 = (b - dot(R,x0)) / D
    x0 = (b - dot(R,x0)) / D
    #Initialize r,v,x (t,s come later)
    r = b - np.dot(A,x0)
    v = r #How do we define this if x_0 is nonzero
    #store list of x values
    xLst = []
    xLst.append(x0)
    x = x0
    ier = 0
    #Iteration
    for its in range(Nmax):
        if (np.linalg.norm(r) <= tol):
            return (x,xLst,its,ier)
        #compute t
        rIP = np.dot(r,r)
        AV = np.dot(A,v)
        t = rIP/np.dot(v,AV)
        #compute x
        x = x + t*v
        xLst.append(x)
        #compute r
        r = r - t*AV
        #compute s
        s = np.dot(r,r)/rIP
        #compute next v
        v = r + s*v
    #unsuccessful
    print("Max Iterations Reached")
    ier = 1
    return (x,xLst,its,ier)

A = np.array([[400000000,2,1,1],[2,-1,1,1],[1,1,1,2],[1,1,2,-1]])
b = np.array([-0.5, 1, 1,4])
guess = np.array([0,0,0,0])
'''
A = array([[2.0,1.0],[5.0,7.0]])
b = array([11.0,13.0])
guess = array([1.0,1.0])
'''


sol = jacobi(A,b,N=25,x=guess)

print("A:")
pprint(A)

print("b:")
pprint(b)

print("x:")
pprint(sol)
