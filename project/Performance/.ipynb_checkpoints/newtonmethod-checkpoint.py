


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product

def Newton(A, b, x0, tol=1e-2):
    xk = x0
    Rk_norm=np.zeros(1000)
    print(xk)
    RHS=inv(A)@(A@xk-b)
    rk_norm = np.linalg.norm(A@xk-b)
    Rk_norm[0]=rk_norm
    error=1.0
    i=0
    while i<100:
        xk=xk-0.1*RHS
        RHS=inv(A)@(A@xk-b)
        print('\nSolution: \t x = {}'.format(xk))
        rk_norm = np.linalg.norm(A@xk-b)
        i=i+1
        print(rk_norm)
        Rk_norm[i]=rk_norm
        #print('Iteration: {} \t x = {} \t residual = {:.8f}'.
              #format(i, xk, rk_norm))
    print('residual = {:.8f}'.format(rk_norm))
    xx=np.linspace(1,i,i) 
    plt.plot(xx,Rk_norm[0:i])
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.grid(True)
A=np.array([[ 1.00665013, -1.41434036, -0.4953264,  -0.43520063],
 [-1.41434036, 3.81637179,  0.58656485,  0.5053143 ],
 [-0.4953264,   0.58656485,  0.87865608,  0.00746661],
 [-0.43520063,  0.5053143,   0.00746661,  0.79932913]] )
b=np.array([-0.99475612,  2.58211651, 0.68135327,  0.56259304])
x_star=np.array([0.5488135,  0.71518937, 0.60276338, 0.54488318])
print('A\n', A, '\n')
print('b\n', b, '\n')
print('The solution x* should be\n', x_star)

x0 = np.array([.4, .6,.9,.5])
xs = Newton(A, b, x0)

#np.allclose(xs[-1], x_star)




# In[ ]:




