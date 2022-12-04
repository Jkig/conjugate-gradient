


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
def LinearCG(A, b, x0, tol=1e-5):
    rk_norm=np.zeros(1000)
    xk = x0
    rk = np.dot(A, xk) - b
    pk = -rk
    rk_norm[0] = np.linalg.norm(rk)
    
    num_iter = 0
    curve_x = [xk]
    i=0
    while rk_norm[i] > tol:
        apk = np.dot(A, pk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        
        num_iter += 1
        if num_iter>999:
            break
        curve_x.append(xk)
        i=i+1
        rk_norm[i] = np.linalg.norm(rk)
        print('Iteration: {} \t x = {} \t residual = {:.8f}'.
              format(num_iter, xk, rk_norm[i]))
    
    print('\nSolution: \t x = {}'.format(xk))
    xx=np.linspace(1,i,i) 
    plt.plot(xx,rk_norm[0:i])
    plt.xlabel('Number of iterations')
    plt.ylabel('Error')
    plt.grid(True)
    return np.array(curve_x)

A=np.array([[ 1.00665013, -1.41434036, -0.4953264,  -0.43520063],
 [-1.41434036, 3.81637179,  0.58656485,  0.5053143 ],
 [-0.4953264,   0.58656485,  0.87865608,  0.00746661],
 [-0.43520063,  0.5053143,   0.00746661,  0.79932913]] )
b=np.array([-0.99475612,  2.58211651, 0.68135327,  0.56259304])
xstar=np.array([0.5488135,  0.71518937, 0.60276338, 0.54488318])
print('A\n', A, '\n')
print('b\n', b, '\n')
print('The solution x* should be\n', x_star)

x0 = np.array([-3, -4,-5,-6])
xs = LinearCG(A, b, x0)

np.allclose(xs[-1], x_star)



# In[ ]:




