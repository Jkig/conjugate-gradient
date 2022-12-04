

import numpy as np

def PCGMethod(A,b,x0,Nmax,tol):
    #Initialize r,v,x (t,s come later)
    r = b - np.dot(A,x0)
    [n1,n2]=A.shape
    xLst = []
    C=np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            if (i == j):
                C[i][j] = A[i][j]

    Sol=np.random.rand(n1)
    x0=np.random.rand(n1)
    r0=b-A@x0
    z0=np.linalg.inv(C)@r0
    d0=z0
    xLst.append(x0)
    ier=0
    #Iteration
    for its in range(Nmax):
        if (np.linalg.norm(r0) <= tol):
            return (x0,xLst,its,ier)
        alpha=np.dot(z0,r0)/np.dot(d0,A@d0)
        x1=x0+alpha*d0
        r1=r0-alpha*A@d0
        z1=np.linalg.inv(C)@r1
        beta=np.dot(z1,r1)/np.dot(z0,r0)
        d1=z1+beta*d0
        d0=d1
        x0=x1
        r0=r1
        z0=z1
        xLst.append(x0)
    #unsuccessful
    print("Max Iterations Reached")
    ier = 1
    return (x0,xLst,its,ier)
