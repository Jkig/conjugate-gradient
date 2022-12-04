import numpy as np

def CGMethod(A,b,x0,Nmax,tol):
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
