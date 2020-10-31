import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv


def Random_Normal_Num():
	random_array = np.array([random.random() for x in range(12)])
	random_num = random_array.sum()-6
	return random_num


def Random_Gauss_Generator(mean, var):
	# Y = (var**1/2) * X + mean
	return (var**(1/2))*Random_Normal_Num() + mean


def Poly_Generator(var, w, phi):
	e = Random_Gauss_Generator(0, var)
	return np.dot(w, phi) + e


def Welford_Online_Algo(x, data_num, pre_mean, pre_var):
	if data_num == 1:
		new_mean = x
		new_var = 0
	else:
		new_mean = pre_mean + (x - pre_mean)/data_num
		new_var = pre_var + ((x - pre_mean)**2)/data_num - pre_var/(data_num-1)
	if data_num>0 and data_num%1000==0 :
		print("The",data_num,"th data is :",x)
		print("mean : ", new_mean, "\tvar : ", new_var, "\n")
	return new_mean, new_var


def Gauss_Online_Learning(data_sum, data_num, pri_mean, pri_var, poly_var):
	var_n = 1/pri_var + data_num/poly_var
	mean_n = var_n * (pri_mean/pri_var + data_sum/poly_var)
	print("mean : ", mean_n, "\tvar : ", var_n, "\n")
	return mean_n, var_n


def plot_line(title, x, y, label, color):
	plt.title(title)
	plt.plot(x, y, label=label, color=color)
	plt.legend(loc = "best")
	plt.show()


def x2phi(x, basis):
	return np.array([x**power for power in range(basis)])


def pred_distribution(mu, phi, a, lambda_):
	mu_pred  = np.matmul(phi.T, mu)
	var_pred = (1/a) + np.matmul(np.matmul(phi.T, inv(lambda_)), phi)
	return mu_pred, var_pred


# def plot_ground_truth_line(x, y):
# 	return plt.plot(x, y, color='green', alpha=0.7, label='Truth')

def plot_pred(index, interval, post_mean, phi, a, post_lam, basis, x_, y_, w_ground):
	xs = np.linspace(-10, 10, 100)
	ys_mu = []
	ys_std_up = []
	ys_std_down = []
	ys_ground = []
	for x in xs:
		phi = x2phi(x, basis)
		mu_pred, var_pred = pred_distribution(post_mean, phi, a, post_lam)
		ys_ground.append(np.matmul(w_ground, phi))
		ys_mu.append(mu_pred)
		ys_std_up.append(mu_pred+var_pred)
		ys_std_down.append(mu_pred-var_pred)
	plt.figure((index+1)/interval)
	title = str(index+1) + " Data trained (1/a = " + str("%.5f" % (1/a)) + ")"
	plt.title(title)
	plt.scatter(x_, y_, marker='.', alpha=0.5)  # Mark each y points we have generated
	plt.plot(xs, ys_mu, color = 'red', label='Guess')  # Plot the guessing line
	plt.plot(xs, ys_ground, color='green', alpha=0.7, label='Truth')  # Plot the ground truth line
	# plot_ground_truth_line(xs, ys_ground)  
	plt.fill_between(xs, ys_std_up, ys_std_down, interpolate=True, color='red', alpha=0.2, label='Predict Distribution')  # Plot the possible distribution
	plt.legend(loc='best')


def Estimate_var(index, xs, ys, w, poly_dim):
	var = 0
	for x,y  in zip(xs, ys):
		phi = x2phi(x, poly_dim)
		mean = np.matmul(w, phi)
		var += (y - mean)**2
	return (1/index) * var


if __name__ == "__main__":

	##### hw-1.a #####
	##### Gaussian Generator #####
	print("\n\n------------HW-1.a------------")
	test_num = 10000
	Y = []
	gaussian_mean = 5
	gaussian_var = 2
	for i in range(test_num):
		Y.append(Random_Gauss_Generator(gaussian_mean, gaussian_var))

	# Plot the hist graph
	x_min = gaussian_mean - 3*(gaussian_var**(1/2))
	x_max = gaussian_mean + 3*(gaussian_var**(1/2))
	plt.title("Result of Gauss\nmean = {}   var = {}".format(gaussian_mean, gaussian_var))
	plt.hist(Y, 100, normed=1, alpha=0.5, range=(x_min, x_max), label='Generated pdf')

	# Plot the ideal gauss line
	x_axis = np.arange(x_min, x_max, 0.1)
	plt.plot(x_axis, norm.pdf(x_axis, gaussian_mean, gaussian_var**(1/2)), label='Ideal Gauss')
	plt.legend(loc='best')
	plt.show()



	##### hw-1.b #####
	##### Ploy Generator #####
	print("\n\n------------HW-1.b------------")
	test_num = 100
	x_range = np.arange(-10, 10, 20/test_num)
	w_ground = [2, 3, 1]  # The weight of the ground true line
	poly_var = 2  # Variance of a point away from the ground true poly line
	poly_dim = len(w_ground)  # The dimension of the ground true poly line

	phi = [[x**power for power in range(poly_dim)] for x in x_range]  # The design matrix

	# Noise Poly
	Y_ideal = []
	Y_noise = []
	for index in range(test_num):
		Y_noise.append(Poly_Generator(poly_var, w_ground, phi[index]))  # Append the noise point of the line into Y_noise
		Y_ideal.append(Poly_Generator(0, w_ground, phi[index]))
	title = "Y = "
	for power, coef in enumerate(w_ground):
		title += "{}x^{}".format(coef, power)
		if power < poly_dim-1:
			title += " + "
	plt.title(title)
	plt.plot(x_range, Y_noise, label="Noise")
	plt.plot(x_range, Y_ideal, label="Ideal")
	plt.legend(loc='best')
	plt.show()



	#### hw-2 #####
	#### Sequential Estimate #####
	print("\n\n------------HW-2------------")
	x = []
	index = 0 # Count the total number of data
	num_data_in_batch = 500

	min_index = 5000
	max_index = 100000
	batch_num = 0 # Count the number of batch
	batch_mean = np.zeros(num_data_in_batch, dtype=np.float32) # Record each mean of a batch
	batch_var = np.zeros(num_data_in_batch, dtype=np.float32) # Record var mean of a batch
	batch_m_mean = [] # Record the mean of mean in each batch
	batch_m_var = [] # Record the mean of var in each batch

	seq_mean = 0 # Count the mean of the online sequence
	seq_var = 0 # Count the var of the online sequence
	total_mean = [] # Record the mean of each result of online algo
	total_var = [] # Record the var of each result of online algo

	while(1):
		x = np.append(x, Random_Gauss_Generator(gaussian_mean, gaussian_var))
		seq_mean, seq_var = Welford_Online_Algo(x[index], index, seq_mean, seq_var)
		total_mean.append(seq_mean)  # Append the seq_mean to the total_mean
		total_var.append(seq_var)  # Append the seq_var to the total_var

		# When the index is larger than 10000, it is time to see if it converges
		# Append every mean and var
		if index > min_index:
			batch_mean = np.append(batch_mean, seq_mean)
			batch_var = np.append(batch_var, seq_var)

		#  And then calculate the mean of the (seq_mean, seq_var) when the batch is full (The size of batch = num_data_in_batch) 
		if index > min_index and index%num_data_in_batch == 0 :
			batch_num += 1
			batch_m_mean.append((batch_mean.sum())/num_data_in_batch)
			batch_m_var.append((batch_var.sum())/num_data_in_batch)

			# Compare the last data (mean, var) to the (mean, var) of last two batch
			if batch_num >= 10 and (abs(seq_mean-batch_m_mean[batch_num-1]) < 0.001 and
									abs(seq_var-batch_m_var[batch_num-1]) < 0.001 and
									abs(seq_mean-batch_m_mean[batch_num-2]) < 0.001 and
									abs(seq_var-batch_m_var[batch_num-2]) < 0.001):
				print("----------------------")
				print("The estimation converges. There are ", index, " data trained.", )
				print("----------------------")
				print("The ground mean : ", gaussian_mean)
				print("Converge mean : ", seq_mean)
				print()
				print("The ground var : ", gaussian_var)
				print("Converge var : ", seq_var)
				print("\n")
				break
			else:
				batch_mean = np.zeros(num_data_in_batch, dtype=np.float32)
				batch_var = np.zeros(num_data_in_batch, dtype=np.float32)
		if index >= max_index-1:
			break
		index += 1
	index += 1
	plot_line("Mean of each result", np.arange(index), total_mean, "mean", "b")
	plot_line("Variance of each result", np.arange(index), total_var, "var", "r")



	##### hw-3 #####
	##### Baysian Linear Regression #####
	print("\n\n------------HW-3------------")
	test_num = 1000
	index = 0
	pic_num = 5
	interval = test_num/pic_num  # Drae a graph every (interval) index
	a = 1  # Guess 1/var of each y, we can assume the first a as any number but 0
	b = 10  # Guess 1/var of weight

	batch_num = 0
	num_data_in_batch = 5
	batch_var = []
	batch_m_var = []

	pri_mean = np.array([0.0 for x in range(poly_dim)])
	pri_lam = np.identity(poly_dim) * b

	x_ = []
	y_ = []

	print("Training...")
	# for index in range(test_num):
	while(1):
		x = np.random.uniform(-10, 10, 1)
		phi = x2phi(x, poly_dim)
		y = Poly_Generator(poly_var, w_ground, phi)

		x_ = np.append(x_, x)
		y_ = np.append(y_, y)

		# Calculate posterior
		post_lam = np.array(a*np.matmul(phi, phi.T) + pri_lam)
		post_mean = np.matmul(inv(post_lam), a*np.matmul(phi, y) + np.matmul(pri_lam, pri_mean))

		# Update the new guessing of 1/var 
		if index > 0:
			a = 1/Estimate_var(index, x_, y_, post_mean, poly_dim)
			batch_var = np.append(batch_var, 1/a)
			# print("a : ", a)
			if index % num_data_in_batch == 0:
				# print("sum : ", batch_var.sum())
				batch_m_var = np.append(batch_m_var, batch_var.sum()/num_data_in_batch)
				batch_var = []
				# print("batch_m_var : ", batch_m_var[batch_num])
				if (abs(1/a-batch_m_var[batch_num-1]) < (0.0005*poly_var) and abs(1/a-batch_m_var[batch_num-2]) < (0.0005*poly_var) ):
					print("----------------------")
					print("The estimation converges. There are ", index+1, " data trained.")
					print("----------------------")
					break
				batch_num += 1

		# Calculate prior
		pri_lam = post_lam
		pri_mean = post_mean

		# Draw a graph every (test_num/pic_num) index
		# if (index+1) % interval == 0:
		# 	print(index+1, "data have been trained...")
		# 	plot_pred(index, interval, post_mean, phi, a, post_lam, poly_dim, x_, y_, w_ground)
		# Draw a graph every index ** 2
		if math.sqrt(index+1) - int(math.sqrt(index+1)) == 0:
			plot_pred(index, interval, post_mean, phi, a, post_lam, poly_dim, x_, y_, w_ground)
			plt.show()
			print(index+1, "data have been trained...")
			print("w_ground : ", w_ground)
			print("w_pred : ", post_mean)
			print()
			print("poly_gaussian_var : ", poly_var)
			print("poly_pred_var : ", 1/a)
			print("\n\n")
		index += 1
	print()
	print("w_ground : ", w_ground)
	print("w_pred : ", post_mean)
	print()
	print("poly_gaussian_var : ", poly_var)
	print("poly_pred_var : ", 1/a)
	plot_pred(index, interval, post_mean, phi, a, post_lam, poly_dim, x_, y_, w_ground)
	plt.show()