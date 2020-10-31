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

def plot_gauss(Y_pred, mean, var):
	x_min = mean - 3*(var**(1/2))
	x_max = mean + 3*(var**(1/2))
	plt.title("Result of Gauss\nmean = {}   var = {}".format(mean, var))
	plt.hist(Y_pred, 100, normed=1, alpha=0.5, range=(x_min, x_max), label='Generated pdf')
	plt.show()

def sigmoid(z):
	if z<-700:
		return 0
	elif z>700 or (z > -0.001 and z < 0.001):
		return 1
	else:
		return 1/(1+math.exp(-z))
	# try:
	# 	return 1/(1+math.exp(-z))
	# except OverflowError:
	# 	print("overflow, -z : ", -z)
	# 	print("overflow, math.exp(-z) : ", math.exp(-z))

def sign(p):
	if p > 0.5:
		return 1
	else:
		return 0

def is_converge(w, count, batch):
	batch_1_mean = np.zeros((1,3))
	batch_2_mean = np.zeros((1,3))

	for i in range(count-batch, count-1):
		batch_1_mean += w[i]
	for i in range(count-batch*2, count-batch-1):
		batch_2_mean += w[i]
	print(batch_2_mean)

	batch_1_mean /= batch
	batch_2_mean /= batch

	if np.sum(w[count]-batch_1_mean)<=1 and np.sum(w[count]-batch_2_mean)<=1 :
		return 1

def compute_loss(X, Y, theta):
    loss = 0
    for x, y in zip(X,Y):
        y_pred = prediction(x, theta)
        l = -1*(y*log(y_pred) + (1-y)*log(1-y_pred))
        if not is_nan(l):
            loss += -1*(y*log(y_pred) + (1-y)*log(1-y_pred))
    if is_nan(loss/len(X)):
        print(loss)
    return loss / len(X)



if __name__ == "__main__":
	
	n = 2000
	counter = 0
	w = np.array([1, 0, 1])

	D1 = []
	D1_x = []
	D1_y = []
	D2 = []
	D2_x = []
	D2_y = []
	X = []
	Y = []

	mx1 = 1
	vx1 = 0.5
	my1 = 6
	vy1 = 7

	mx2 = 10
	vx2 = 1
	my2 = 3
	vy2 = 5

	for index in range(n):
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

	# X is the design matrix
	X = np.asarray(X).reshape(2*n, 3)
	Y = np.array(Y)

	# xs records whole x points
	xs = []
	xs = np.append(xs, D1_x)
	xs = np.append(xs, D2_x)


	while(1):
		print("counter : ", counter)
		Y_sigmoid = []
		# Y_pred = []
		loss = 0
		D = np.zeros((2*n, 2*n))
		for i in range(2*n):
			print("X[", i, "]", X[i])
			print("w : ", w)
			z = np.dot(X[i], w)
			print("z : ", z)
			# tmp_sigmoid = sigmoid(z)
			# print("tmp_sigmoid : ", tmp_sigmoid)
			Y_sigmoid.append(sigmoid(z))
			# print("sign(tmp_sigmoid) : ", sign(tmp_sigmoid))
			# Y_pred.append(i%2)
			# print(Y_pred)
			# Y_pred.append(sign(tmp_sigmoid))
			# print()

			try :
				loss += -1*(Y[i]*log(Y_sigmoid[i]) + (1-Y[i])*log(1-Y_sigmoid[i]))
			except ValueError:
				print("valueerror")
				continue

			try:
				D[i][i] = math.exp(-z) / ((1+math.exp(-z))**2)
			except OverflowError:
				print("overflow")
				D[i][i] = 0
		# print(Y)
		print("loss : ", loss)
		print()
		Y_sigmoid = np.array(Y_sigmoid)
		# print("Y_sigmoid :", (Y_sigmoid))
		# print("Y_pred :", (Y_pred))
		print("D : \n", D)
		print()
		print("X : \n", X)
		print()
		print("X.T : \n", X.T)
		print()
		print("np.matmul(X.T, D) : \n", np.matmul(X.T, D))
		print()
		print("np.matmul(np.matmul(X.T, D), X) : \n", np.matmul(np.matmul(X.T, D), X))
		print()
		Hession = np.matmul(np.matmul(X.T, D), X)
		print("Hession : \n", Hession)
		print()
		print("Y_sigmoid : \n", Y_sigmoid)
		print()
		print("Y_sigmoid-Y : \n", Y_sigmoid-Y)
		print()
		gradient = np.matmul(X.T, (Y_sigmoid-Y).reshape(2*n, 1))
		print("gradient : \n", gradient)

		# print("(np.matmul(np.linalg.inv(Hession),gradient).reshape(1,3)) : \n", (np.matmul(np.linalg.inv(Hession),gradient).reshape(1,3)))
		if linalg.cond(Hession) < 1/sys.float_info.epsilon:
			print("np.linalg.inv(Hession) : \n", np.linalg.inv(Hession))
			tmp_update = 0.1 * np.squeeze((np.matmul(np.linalg.inv(Hession),gradient).reshape(1,3)))
		else:
			tmp_update = 0.1 * gradient
		print("tmp_update : \n", tmp_update)
		w = w - tmp_update
		w = np.squeeze(w)
		print("w : ", w)
		print()

		if loss <= 0.01:
			break

		counter += 1
		print("----------------------\n\n")

		if counter%5==0:
			x_range = np.arange(np.amin(xs)-0.3*vx1, np.amax(xs)+0.3*vx1, 0.1)
			plt.scatter(D1_x, D1_y, marker='.', alpha=0.5, color='blue')
			plt.scatter(D2_x, D2_y, marker='.', alpha=0.5, color='red')
			plt.plot(x_range, -(w[0]+w[1]*x_range)/w[2])
			plt.show()
			print("\nw : ", w)
			print("\ngradient : \n", gradient)
			print("\n---------------------")

	x_range = np.arange(np.amin(xs)-0.3*vx1, np.amax(xs)+0.3*vx1, 0.1)
	plt.scatter(D1_x, D1_y, marker='.', alpha=0.5, color='blue')
	plt.scatter(D2_x, D2_y, marker='.', alpha=0.5, color='red')
	plt.plot(x_range, -(w[0]+w[1]*x_range)/w[2])
	plt.show()
	print(counter, "Data have been trained...")
	print("\nw : ", w)
	print("\ngradient : \n", gradient)
	print("\nThe result has Converged.")