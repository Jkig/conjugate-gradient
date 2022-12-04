import matplotlib.pyplot as plt
from CGMethod import CGMethod
import numpy as np
# make sure you have installed ssgetpy
import ssgetpy

# to read from file into np.array
from io import StringIO
from scipy.io import mmread



def Driver():
    # stuff here pulls matricies
    #'''
    n = (1900, 2000)
    a = ssgetpy.search(rowbounds = n, colbounds = n, dtype = 'real', isspd = True, limit = 1)[0]
    a.download(destpath = 'huge-matrix')
    # print(a)
    return
    #'''
    # stuff here takes a .mtx file and returns a numpy array
    f = open("matrix4.mtx", "r", encoding="utf-8")
    text = f.read()
    # print(text)
    m = mmread(StringIO(text))
    m = m.todense()
    # print(m)

    '''
    # some testing stuff, getting wierded out by SPD
    x = np.random.randint(0,10,14)
    y = np.matmul(np.matmul(np.transpose(x), m), x)
    print(x)
    print(y)
    '''
    


Driver()

