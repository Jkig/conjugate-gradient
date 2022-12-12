import matplotlib.pyplot as plt
from CGMethod import CGMethod
import numpy as np
from numpy.linalg import inv

from io import StringIO
from scipy.io import mmread

def Block_Jacobi(A, block_size=2):
    # I'm starting to do it with just 2x2
    # maybe i can have thie first maybe be a 3x3??
    
    # take in og matrix, and return the preconditioner, ready to multiply
    Pi = np.zeros([A[0].size, A[0].size])

    # make a loop that keeps grabing and making sub-matricies

    sub_p = np.zeros([2,2])
    i = 0
    for i in range(0,int(A[0].size/2)):
        sub_p[0,0] = A[2*i,2*i]
        sub_p[0,1] = A[2*i,2*i+1]
        sub_p[1,0] = A[2*i+1,2*i]
        sub_p[1,1] = A[2*i+1,2*i+1]
        
        sub_pi = inv(sub_p)
        
        Pi[2*i,2*i] = sub_pi[0,0]
        Pi[2*i,2*i+1] = sub_pi[0,1]
        Pi[2*i+1,2*i] = sub_pi[1,0]
        Pi[2*i+1,2*i+1] = sub_pi[1,1]
    i+=1
    if ((A[0].size)%2) == 1:
        Pi[2*i,2*i] = (1/(A[2*i,2*i]))

    '''    
    print()
    print(Pi)
    print("A:-----")
    print(A)
    print(np.matmul(Pi,A))
    '''
    print("the post pre-conditioned matrix")
    print(np.linalg.cond(np.matmul(Pi,A)))
    return Pi


def driver(A,b,x0, do_precon):
    #Set exit parameters
    Nmax = 1500
    tol = 1.0e-8
    #initial condition

    # precondition - do i put in the p*a, x0, p*b?? i'll try this, runtime can tell me
    #   if this doesn't work
    Pi = np.identity(A[0].size)
    if do_precon == True:
        Pi = Block_Jacobi(A)

    
    #run evaluator
    x,xLst,its,ier = CGMethod(np.matmul(Pi,A),np.matmul(Pi,b),x0,Nmax,tol)
    # print("x = ", x)
    print("Number of Iterations: ", its)

    '''plotting'''
    # list of abs errors vs last one
    abs_err = []
    last_index = len(xLst) - 1
    '''
    for i in range(len(xLst)):
        error = np.linalg.norm(xLst[i]-xLst[last_index])
        abs_err.append(error)
    '''
    # this also calculates with gaussian, so i can see how far off non-convergent ones are
    real_sol = np.linalg.solve(A,b)
    
    for i in range(len(xLst)):
        # error = np.linalg.norm(xLst[i]-xLst[last_index])
        error = np.linalg.norm(xLst[i]-real_sol)
        abs_err.append(error)
    
    plt.plot(range(0,len(xLst)),abs_err)
    plt.show()
    



if __name__ == '__main__':
    A = np.identity(5)
    A = 2*A
    b = np.random.rand(5)
    x0 = np.zeros(5)

    driver(A, b, x0, True)


    
    
    # good test:
    print("good test")
    
    # with no pre-conditioning:
    #   this doesn't even converge in 2900 iterations lol
    f = open("over200.mtx", "r", encoding="utf-8")
    text = f.read()
    m = mmread(StringIO(text))
    A = m.todense()
    A = np.array(A)
    
    b = np.random.rand(237)
    x0 = np.zeros(237)
    print()
    print("big matrix, NOT - preconditioned, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A, b, x0, False)

    # try it all with preconditioned:
    print()
    print("big matrix, preconditioned, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A, b, x0, True)
    
    '''

    # trying to find more decent examples:
    # block jacobi blows up here
    f = open("662_bus.mtx", "r", encoding="utf-8")
    text = f.read()
    m = mmread(StringIO(text))
    A = m.todense()
    A = np.array(A)
    
    b = np.random.rand(662)
    x0 = np.zeros(662)
    print()
    print("662_bus, NOT - preconditioned, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A, b, x0, False)

    # try it all with preconditioned:
    print()
    print("662_bus, preconditioned, condition number:", np.linalg.cond(A), "size of matrix: ", len(A))
    driver(A, b, x0, True)
    '''

