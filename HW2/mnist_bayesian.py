import numpy as np
import scipy
import scipy.stats
import time
import struct


def read_label(file):
	data = open(file, 'rb').read()

	offset = 0
	data_format = ">ii"
	magic_num, item_num = struct.unpack_from(data_format, data, offset)
	# print("Magic num : {}\nNum of item : {}".format(magic_num, item_num))

	label = np.empty(item_num)
	offset = struct.calcsize(data_format)
	data_format = ">B"
	for i in range(item_num):
		label[i] = struct.unpack_from(data_format, data, offset)[0]
		offset += struct.calcsize(data_format)
	# label = map(float, label)
	return item_num, label



def read_image(file):
	data = open(file, 'rb').read()

	offset = 0
	data_format = ">iiii"
	magic_num, im_num, row, col = struct.unpack_from(data_format, data, offset)

	im_size = row*col
	# image = np.empty((im_num, row, col))
	image = np.empty((im_num,im_size))
	offset = struct.calcsize(data_format)
	data_format = ">" + str(im_size) + "B"
	for i in range(im_num):
		# Let the size of image be (num * pixels)
		image[i] = np.array(struct.unpack_from(data_format, data, offset))
		offset += struct.calcsize(data_format)
		# print image[i]
	image = image.astype(np.float)
	return im_num, im_num, im_size, image


def __log_prior(label_num, train_label, train_num):
	prior = np.zeros((label_num), dtype = np.float)

	for x in range(train_num):
		prior[train_label[x]] += 1

	return np.log(prior/train_num)


def Discrete_log_LH(label_num, pixel_num, train_label, train_image, train_num):
	LH = np.ones((label_num, pixel_num, 32))
	LH *= 10  # Prevent LH from becomimg 0 after taking log

	print("Training Data.....")
	for x in range(train_num):
		if x % 5000 == 0:
			print("{} Data has been trained".format(x))
		for y in range(pixel_num):
			LH[train_label[x]][y][train_image[x][y]] += 1

	for x in range(label_num):
		for y in range(pixel_num):
			appear_count = np.sum(LH[x][y])
			LH[x][y] /= appear_count

	return np.log(LH)



def Discrete_predict(label_num, pixel_num, log_LH, log_pri, test_image, test_label, test_num):
	
	f = open("Discrete_Result.txt", "w")

	discrete_res = np.zeros(test_num)
	correct_count = 0

	print("\n\nPredicting Data.....")
	for z in range(test_num):
		if z % 5000 == 0:
			print("{} Data has been predicted".format(z))
		discrete_posterior = np.zeros(label_num)
		for x in range(label_num):
			for y in range(pixel_num):
					discrete_posterior[x] += log_LH[x][y][test_image[z][y]]
		discrete_posterior += log_pri

		f.write("Posterior of test data {} is : \n{}\n\n".format(z, discrete_posterior))

		discrete_res[z] = np.argmax(discrete_posterior)

		if discrete_res[z] == test_label[z]:
			correct_count += 1

	correct_percent = 100 * float(correct_count) / float(test_num)

	print("\nTest Prediction : {}\n".format(discrete_res))
	print("Test Labels are : {}\n".format(test_label[:test_num]))
	print("Correct % : {}%\n".format(correct_percent))
	f.write("\nTest Prediction : {}\n\n".format(discrete_res))
	f.write("Test Labels are : {}\n\n".format(test_label[:test_num]))
	f.write("Correct % : {}%".format(correct_percent))
	f.close()


def Gauss_train(label_num, pixel_num, train_image, train_label, train_num):

	# The appear times of each label
	appear_count = np.zeros(label_num)
	for x in range(train_num):
		appear_count[train_label[x]] += 1

	# The mean of every pixel for each label
	mean = np.zeros((label_num, pixel_num))
	for x in range(label_num):
		for y in range(train_num):
			if train_label[y] == x:
				mean[x] += train_image[y]
		mean[x] /= appear_count[x]

	# The var of every pixel for each label
	var = np.zeros((label_num, pixel_num))
	for x in range(label_num):
		for y in range(train_num):
			if train_label[y] == x:
				var[x] += (train_image[y]-mean[x])**2
		var[x] /= appear_count[x]
	return mean, var


def Gauss_log_LH_sum(label_num, pixel_num, single_test_image, mean, var):

	LH = 0
	for x in range(pixel_num):
		LH += scipy.stats.norm(mean[x], var[x]).logpdf(single_test_image[x])
	return LH
 

def Continuous_predict(label_num, pixel_num, log_pri, test_image, test_label, test_num, mean, var):

	f = open("Continuous_Result.txt", "w")

	continuous_res = np.zeros(test_num)
	correct_count = 0

	var += 0.01

	for z in range(test_num):
		continuos_posterior = np.zeros(label_num)
		for x in range(label_num):
			log_LH_sum = Gauss_log_LH_sum(label_num, pixel_num, test_image[z], mean[x], var[x])
			continuos_posterior[x] = log_pri[x] + log_LH_sum

		f.write("Posterior of test data {} is : \n{}\n\n".format(z, continuos_posterior))

		continuous_res[z] = np.argmax(continuos_posterior)

		if continuous_res[z] == test_label[z]:
			correct_count += 1

		if z%100 == 0 and z!=0 :
			print("\n{} Data has been predicted.....".format(z))
			print("Correct % : {}%\n".format(100 * float(correct_count)/float(z+1)))
			f.write("Correct % : {}%\n".format(100 * float(correct_count)/float(z+1)))

	correct_percent = 100 * float(correct_count) / float(test_num)

	print("Test Prediction : {}\n".format(continuous_res))
	print("Test Labels are : {}\n".format(test_label[:test_num]))
	print("Correct % : {}%\n".format(correct_percent))
	f.write("\nTest Prediction : {}\n\n".format(continuous_res))
	f.write("Test Labels are : {}\n\n".format(test_label[:test_num]))
	f.write("Correct % : {}%".format(correct_percent))
	f.close()


def gamma(x):
	if x == 1 or x == 2:
		return 1
	else:
		return (x-1) * gamma(x-1)

def beta(theta, a, b):
	return theta**(a-1) * (1-theta)**(b-1) * gamma(a+b) / gamma(a) / gamma(b)

def combination(m, n):
	if n>(m/2):
		n = m-n

	upper = 1
	for i in range(n):
		upper *= (m-n)

	lower = gamma(n+1)
	return upper/lower


def online_learning(data, a, b):

	theta = float(a)/float(a+b)

	for x in range(len(data)):
		m = 0
		N = 0
		for y in range(len(data[x])):
			N += 1
			if data[x][y] == '1':
				m += 1

		prior = beta(theta, a, b)
		likelihood = combination(N, m) * theta**m * (1-theta)**b
		theta = float(a)/float(a+b)
		posterior = beta(theta, a+m, b+N-m)

		print("Data {} : {}".format(x, data[x]))
		print("theta is : {}\nLikelihood : {}\tPrior : {}\tPosterior : {}\n".format(theta, likelihood, prior, posterior))
		a += m
		b += N-m


def run():

	train_label_file = "train-labels-idx1-ubyte"
	train_image_file = "train-images-idx3-ubyte"
	train_label_num, train_label = read_label(train_label_file)
	train_image_num, image_row, image_col, train_image = read_image(train_image_file)

	test_label_file = "t10k-labels-idx1-ubyte"
	test_image_file = "t10k-images-idx3-ubyte"
	test_label_num, test_label = read_label(test_label_file)
	test_image_num, test_row, test_image, test_image = read_image(test_image_file)

	train_label = train_label.astype(int)
	train_image = train_image.astype(int)
	test_image = test_image.astype(int)

	tmp_train_image = np.asarray([train_image[row][:] for row in range(len(train_image))])
	tmp_test_image = np.asarray([test_image[row][:] for row in range(len(test_image))])

	label_num = 10
	pixel_num = 784

	### Continuous Method ###
	print("\n\n\n------ Continuous Method ------")
	# limit = 1000
	limit = test_image_num
	train_image = train_image.astype(np.float32)/255.0
	test_image = test_image.astype(np.float32)/255.0

	mean, var = Gauss_train(label_num, pixel_num, train_image, train_label, image_row)
	continuous_log_prior = __log_prior(label_num, train_label, train_label_num)
	Continuous_predict(label_num, pixel_num, continuous_log_prior, test_image[:limit][:], test_label[:], limit, mean, var)


	### Discrete Method ###
	print("------ Discrete Method ------")
	tmp_train_image = tmp_train_image.astype(np.int) / 8
	tmp_test_image = tmp_test_image.astype(np.int) / 8
	# limit = 1000
	limit = test_image_num
	discrete_log_prior = __log_prior(label_num, train_label, train_label_num)
	discrete_log_LH = Discrete_log_LH(label_num, pixel_num, train_label, tmp_train_image, train_label_num)
	Discrete_predict(label_num, pixel_num, discrete_log_LH, discrete_log_prior, tmp_test_image[:limit][:], test_label[:limit], limit)


	### Online learning ###
	print("------ Online Learning ------")
	file = open("coin_data.txt")
	data = file.read().splitlines()
	file.close()

	prior_a = input("Please enter a : ")
	prior_b = input("Please enter b : ")
	print("")

	online_learning(data, prior_a, prior_b)


if __name__ == "__main__":
	start = time.clock()
	run()
	end = time.clock()
	print("It costs {} seconds.".format(end - start))