#%%
import pandas as pd
df = pd.read_csv("/home/airplaneless/source/repos/somestuff/MachineLearningCoursera/3 week/data.csv", header=None)
X1 = df[1]
X2 = df[2]
Y = df[0]

#%%
from math import exp
from tqdm import tqdm

k = 0.5
C = 1
err = 10**-5

def adjust_w1(Y, X1, X2, w):
    s = 0
    for i in range(len(Y)):
        s += Y[i]*X1[i]*(1 - 1/(1 + exp(-Y[i]*(w[0]*X1[i] + w[1]*X2[i]))))
    s = s*k/len(Y)
    return s - k*C*w[0]

def adjust_w2(Y, X1, X2, w):
    s = 0
    for i in range(len(Y)):
        s += Y[i]*X2[i]*(1 - 1/(1 + exp(-Y[i]*(w[0]*X1[i] + w[1]*X2[i]))))
    s = s*k/len(Y)
    return s - k*C*w[1]

def grad_desc(Y, X1, X2):
    w = list([0, 0])
    for i in tqdm(xrange(100000)):
        adj1 = adjust_w1(Y, X1, X2, w)
        adj2 = adjust_w1(Y, X1, X2, w)
        new1 = w[0] + adj1
        new2 = w[1] + adj2
        if (new1 - w[0])**2 + (new2 - w[1])**2 <= err:
            return w
        else:
            pass
    raise ValueError

#%%
w = grad_desc(Y, X1, X2)
w

#%%
w