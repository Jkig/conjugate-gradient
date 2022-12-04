

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product

def GD(A, b, x0, tol=1e-5):
    xk = x0
    rk_norm=np.zeros(1000)
    RHS=A@xk-b
    rk = np.dot(A, xk) - b
    rk_norm[0] = np.linalg.norm(rk)
    #print(rk_norm[0])
    i=0
    while i<50:
        xk=xk-0.3*RHS
        RHS=A@xk-b
        print('\nSolution: \t x = {}'.format(xk))
        i=i+1
        rk_norm[i] = np.linalg.norm(RHS)
        #print(rk_norm[i])
        
        print('Iteration: {} \t x = {} \t residual = {:.8f}'.
              format(i, xk, rk_norm[i]))
    xx=np.linspace(1,i,i) 
    plt.plot(xx,rk_norm[0:i])
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.grid(True)
    #print(rk_norm)

np.random.seed(0)
A=np.array([[ 1.00665013, -1.41434036, -0.4953264,  -0.43520063],
 [-1.41434036, 3.81637179,  0.58656485,  0.5053143 ],
 [-0.4953264,   0.58656485,  0.87865608,  0.00746661],
 [-0.43520063,  0.5053143,   0.00746661,  0.79932913]] )
b=np.array([-0.99475612,  2.58211651, 0.68135327,  0.56259304])
x_star=np.array([0.5488135,  0.71518937, 0.60276338, 0.54488318])

print('A\n', A, '\n')
print('b\n', b, '\n')
print('The solution x* should be\n', x_star)

x0 = np.array([.4, .6,.7,.8])
xs = GD(A, b, x0)

#np.allclose(xs[-1], x_star)





# In[ ]:




