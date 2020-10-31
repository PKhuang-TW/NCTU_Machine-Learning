import numpy as np
import struct
import math

def read_label(file):
	data = open(file, 'rb').read()

	offset = 0
	data_format = ">ii"
	magic_num, item_num = struct.unpack_from(data_format, data, offset)

	label = np.empty(item_num)
	offset = struct.calcsize(data_format)
	data_format = ">B"
	for i in range(item_num):
		label[i] = struct.unpack_from(data_format, data, offset)[0]
	offset += struct.calcsize(data_format)
	return item_num, label



def read_image(file):
	data = open(file, 'rb').read()

	offset = 0
	data_format = ">iiii"
	magic_num, im_num, row, col = struct.unpack_from(data_format, data, offset)

	im_size = row*col
	image = np.empty((im_num,im_size))
	offset = struct.calcsize(data_format)
	data_format = ">" + str(im_size) + "B"
	for i in range(im_num):
		# Let the size of image be (num * pixels)
		image[i] = np.array(struct.unpack_from(data_format, data, offset))
		offset += struct.calcsize(data_format)
		# print image[i]
	image = image.astype(np.float)
	return im_num, im_num, image


if __name__ == "__main__":
	train_label_file = "train-labels-idx1-ubyte"
	train_image_file = "train-images-idx3-ubyte"
	train_label_num, train_label = read_label(train_label_file)
	train_image_num, image_row, train_image = read_image(train_image_file)

	test_label_file = "t10k-labels-idx1-ubyte"
	test_image_file = "t10k-images-idx3-ubyte"
	test_label_num, test_label = read_label(test_label_file)
	test_image_num, test_row, test_image = read_image(test_image_file)

	train_image = np.floor(train_image / 128)
	test_image = np.floor(test_image / 128)

	train_count = np.zeros(10)
	test_count = np.zeros(10)
	for i in range(10):
		train_count[i] = np.count_nonzero(train_label==i)
		test_count[i] = np.count_nonzero(test_count==i)