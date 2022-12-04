


import numpy as np
def gaussy(A, b, n):
    l = [0 for x in range(n)]
    s = [0.0 for x in range(n)]
    for i in range(n):
        l[i] = i
        smax = 0.0
        for j in range(n):
            if abs(A[i][j]) > smax:
                smax = abs(A[i][j])
        s[i] = smax

    for i in range(n - 1):
        rmax = 0.0
        for j in range(i, n):
            ratio = abs(A[l[j]][i]) / s[l[j]]
            if ratio > rmax:
                rmax = ratio
                rindex = j
        temp = l[i]
        l[i] = l[rindex]
        l[rindex] = temp
        for j in range(i + 1, n):
            multiplier = A[l[j]][i] / A[l[i]][i]
            for k in range(i, n):
                A[l[j]][k] = A[l[j]][k] - multiplier * A[l[i]][k]
            b[l[j]] = b[l[j]] - multiplier * b[l[i]]

    x = [0.0 for y in range(n)]
    x[n - 1] = b[l[n - 1]] / A[l[n - 1]][n - 1]
    for j in range(n - 2, -1, -1):
        summ = 0.0
        for k in range(j + 1, n):
            summ = summ + A[l[j]][k] * x[k]
        x[j] = (b[l[j]] - summ) / A[l[j]][j]

    print ("The solution vector using Gauss Elimination is [", end="")
    for i in range(n):
        if i != (n - 1):
            print(x[i], ",", end="")
        else:
            print(x[i], "].")

A=np.array([[ 1.00665013, -1.41434036, -0.4953264,  -0.43520063],
 [-1.41434036, 3.81637179,  0.58656485,  0.5053143 ],
 [-0.4953264,   0.58656485,  0.87865608,  0.00746661],
 [-0.43520063,  0.5053143,   0.00746661,  0.79932913]] )
b=np.array([-0.99475612,  2.58211651, 0.68135327,  0.56259304])
x_star=np.array([0.5488135,  0.71518937, 0.60276338, 0.54488318])
print('The solution is:')
print(x_star)
gaussy(A, b, 4)


# In[ ]:




