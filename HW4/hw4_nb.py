import random
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import diff
from math import log
from numpy import linalg

def Random_Normal_Num():
    random_array = np.array([random.random() for x in range(12)])
    random_num = random_array.sum()-6
    return random_num

def Random_Gauss_Generator(mean, var):
    return (var**(1/2))*Random_Normal_Num() + mean

def sigmoid(z):
    try:
        return 1/(1+math.exp(-z))
    except OverflowError:
        return overflow_determine(-z)

def draw_pred(w, Gradient, counter):
    x_range = np.arange(np.amin(xs)-0.3*vx1, np.amax(xs)+0.3*vx1, 0.1)
    plt.scatter(D1_x, D1_y, marker='.', alpha=0.5, color='blue')
    plt.scatter(D2_x, D2_y, marker='.', alpha=0.5, color='red')
    plt.plot(x_range, -(w[0]+w[1]*x_range)/w[2])
    plt.show()
    print("%d data have been trained" %counter)
    print("\nw : ", w)
#     print("\nGradient : \n", Gradient)
    print("loss : ", loss)
    print("---------------------\n")

def overflow_determine(x):
    print("OVERFLOWEXCEPTION EXCEPTION")
    f_MAX = sys.float_info.max
#     f_MIN = sys.float_info.min
    if x > 0:
        return f_MAX
    else:
        return 0

def hession(X, z, rows):
    D = np.zeros((rows, rows))
    for i in range(rows):
        try:
            tmp_e = math.exp(-z[i])
        except OverflowError:
            tmp_e = overflow_determine(-z[i])
            
        if tmp_e>=1.e+5 or tmp_e<=-1.e+5:
            D[i][i] = 0
        elif tmp_e<=0.00001 and tmp_e>=-0.00001:
            D[i][i] = 1
        else:
            D[i][i] = tmp_e / ((1+tmp_e**2))
    return np.matmul(np.matmul(X.T, D), X)


def gradient(X, Y, z, rows):
    Y_sigmoid = []
    loss = 0
    for i in range(rows):
        tmp_sigmoid = sigmoid(z[i])
        Y_sigmoid.append(tmp_sigmoid)
        loss += cal_loss(Y[i], tmp_sigmoid)
    return loss/rows, np.matmul(X.T, (Y_sigmoid-Y).reshape(rows, 1))


def cal_loss(Y, Y_pred):
    try :
        return -1*(Y*log(Y_pred) + (1-Y)*log(1-Y_pred))
    except ValueError:
        return 0


data_num = 1000
mat_row = 2*data_num

mx1 = 0
my1 = 0
vx1 = 1
vy1 = 1

mx2 = 5
my2 = 5
vx2 = 1
vy2 = 1

D1=[]
D1_x=[]
D1_y=[]

D2=[]
D2_x=[]
D2_y=[]

X=[]
Y=[]

for i in range(data_num):
    x = Random_Gauss_Generator(mx1, vx1)
    y = Random_Gauss_Generator(my1, vy1)
    D1_x.append(x)
    D1_y.append(y)
    D1.append((x, y))
    X.append([1, x, y])
    Y.append(0)

    x = Random_Gauss_Generator(mx2, vx2)
    y = Random_Gauss_Generator(my2, vy2)
    D2_x.append(x)
    D2_y.append(y)
    D2.append((x, y))
    X.append([1, x, y])
    Y.append(1)

X = np.asarray(X).reshape(mat_row, 3)
Y = np.array(Y)

# xs records whole x points
xs = []
xs = np.append(xs, D1_x)
xs = np.append(xs, D2_x)

w = np.array([0,0,0])
lr = 0.1
counter = 0
# print(z)

while(1):
    z = np.matmul(X, w.T)
    Hession = hession(X, z, mat_row)
    loss, Gradient = gradient(X, Y, z, mat_row)
    
    if linalg.cond(Hession) < 1/sys.float_info.epsilon: 
        update = np.matmul(np.linalg.inv(Hession), Gradient).reshape(1,3)
        update = np.squeeze(update)
    else:
        update = Gradient

    w = w-update
    counter += 1
    
    if loss<=0.002:
        break
    elif counter>=5000:
        break
    
    if counter%50==0:
        print("%d data have been trained..." %counter)
    
    if counter%100==0:
        draw_pred(w, Gradient, counter)
        
draw_pred(w, Gradient, counter)

print("The result has Converged.")
w /= w[2]
# print("\nThe final line is : ")
# print("Y = %.2fX + %.2f" %(w[1], w[0]))