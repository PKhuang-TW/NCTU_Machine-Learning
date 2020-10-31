# Function to print the result properly for python 2.7
from __future__ import print_function


# Identity Matrix
def identity(size):
    matrix = [[0]*size for i in range(size)]
    for i in range(size):
        matrix[i][i] = 1
    return matrix

# Matrix nultiply constant
def Mat_mul_constant(mat, constant):
	if type(mat[0]) is not type([1]):
		return [x*constant for x in mat]
	else:
		count_x = 0
		for x in mat:
			count_y = 0
			for y in x:
				mat[count_x][count_y] = float(y*constant)
				count_y += 1
			count_x += 1
		return mat

# Matrix transpose
def transpose(mat):
	return [list(x) for x in zip(*mat)]

# Matrix multiplu Matrix
def Mat_mul(A, B):
	return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)]	for A_row in A] 

# Matrix add Matrix
def Mat_addition(X,Y):
	return [[X[i][j] + Y[i][j]  for j in range(len(X[0]))] for i in range(len(X))]


# Reduce every row from the pivot row
def reduce_row(piv_row, tmp_row, Mat_U, Mat_L):
	for x in range(0, len(tmp_row)):
		# Calculate the scale difference between pivot row and every extra row
		scale = tmp_row[x][piv_row] / Mat_U[piv_row][piv_row]
		# print("scale is : {}".format(scale))

		# After reducing extra row from the pivot row, append it in Mat_U
		Mat_U.append(map(lambda (a,b):a-scale*b, zip(tmp_row[x], Mat_U[piv_row])))

		# The scale we've done should also append it in Mat_L
		Mat_L[x+piv_row+1][piv_row] = scale
	# print("pivot is : {}, Mat_U is : {}".format(piv_row, Mat_U))


# Calculate the extra rows in Mat_U, and pop it into tmp_row
def extra_row(piv_row, tmp_row, M, Mat_U, Mat_L):
	# initiate the first tmp_row
	if piv_row==0:
		tmp_row = M[1:]

	# every row below the pivot row is an extra row
	else:
		for x in range(0,final_rows - piv_row - 1):
			tmp_row.insert(0, Mat_U.pop())
	# print("\n\npivot is {}, tmp_row is : {}".format(piv_row, tmp_row))
	# print("Mat_U in extra_row is : {}".format(Mat_U))
	return tmp_row


# LU function
def LU_decomp(M, B):
	Mat_U = []
	Mat_L = identity(final_rows)
	Mat_U.append(M[0])  # initiate the first row of Mat_U

	# reduce every row and exchange the pivot row
	for x in range(0, final_rows-1):
		tmp_row = [];
		reduce_row(x, extra_row(x, tmp_row, M, Mat_U, Mat_L), Mat_U, Mat_L)


	print("\nB is : {}".format(B))
	print("\nA_U is : {}".format(Mat_U))
	print("\nA_L is : {}".format(Mat_L))

	# Multiplying AT and b
	# print("Mat_U is : {}".format(Mat_U))
	# print("Mat_L is : {}\n\n".format(Mat_L))

	tmp_sol = []
	sol = []

	# Solve y in Ly=b
	tmp_sol.append([B[0][0]/Mat_L[0][0]])
	# print("tmp_sol is : {}".format(tmp_sol))
	for x in range(1, final_rows):
		tmp = B[x][0]
		# print("\nx is {}, and tmp is {}".format(x, tmp))
		for y in range(0, x):
			# print("Mat_L[x][y] is {}".format(Mat_L[x][y]))
			tmp -= Mat_L[x][y]*tmp_sol[y][0]
		# print("tmp is {}".format(tmp))
		tmp_sol.append([tmp])
	# print tmp_sol


	# Solve x in Ux=y
	sol.append([tmp_sol[final_rows-1][0]/Mat_U[final_rows-1][final_rows-1]])
	for x in range(0, final_rows-1)[::-1]:
		tmp = tmp_sol[x][0]
		# print("x is {}, and tmp is {}".format(x, tmp))
		for y in range(x+1, final_rows):
			# print("Mat_U[x][y] is {}".format(Mat_U[x][y]))
			# print("sol[y-x-1][0] is {}".format(sol[y-x-1][0]))
			tmp -= Mat_U[x][y]*sol[y-x-1][0]
		sol.insert(0, [tmp/Mat_U[x][x]])
	return sol
	


def Newton(Mat, B):
	x_n = [[0] for x in range(0, final_rows)]
	# print("x_n is : {}".format(x_n))
	Hession = Mat_mul_constant(ATA, 2)
	print("\nHession is : {}".format(Hession))
	# print("\nMat_mul(Hession, x_n) is : {}".format(Mat_mul(Hession, x_n)))
	Gradient = Mat_addition(Mat_mul(Hession, x_n), Mat_mul_constant(B, -2))
	print("\nGradient is :{}".format(Gradient))
	# print("\nLU return : {}".format(LU_decomp(Hession, Gradient)))
	return Mat_addition(Mat_mul_constant(LU_decomp(Hession, Gradient), -1), x_n)



########
# Main #
########

A = []
data = []

dim = input("\nWhat is the dimension you'd like to have?\n")
lam = input("What is the value of lambda you'd like to have?\n")

# Open the data file
file = open("data.txt") 
data_lines = 0
while 1:
	lines = file.readlines()
	if not lines:
		break
	for line in lines:
		data_lines += 1
		for comma in range(1,len(line)):
			if line[comma] is ",":
				tmp_list = []
				for x in range(0,dim):
					tmp_list.append(float(line[0:comma])**x)
				A.append(tmp_list)
				if line is not lines[-1]:
					data.append([float(line[comma+1:-1])])
				else:
					data.append([float(line[comma+1:])])
file.close()


# Print the matrix A and data
print("\nA is : ")
for x in range(0, len(A)):
	print("{}".format(A[x]))

print("\ndata is : ")
for x in range(0, len(data)):
	print("{}".format(data[x]))


# Counting the numbers of rows in AT*A
final_rows = dim
print("final_rows is : {}".format(final_rows))

# Calculating AT*A
A_T = transpose(A)
ATA = Mat_mul(A_T, A)
print("\nATA is : {}".format(ATA))


# Calucula]ting AT*A+lambda*I
Mat = [ ATA[row][:] for row in range(len(ATA))]
for x in range(0,final_rows):
	Mat[x][x] += lam
print("\nMat is : {}".format(Mat))

B = Mat_mul(A_T, data)

LSE_sol = LU_decomp(Mat, B)


# Print LSE solution
print("\nThe LSE solution is :\ny = ", end="")
for x in range(0, final_rows):
	print("{} x^{}".format(LSE_sol[x][0], x), end=" ")
	if x is not final_rows-1:
		print("+", end=" ")
print("")

 # Print LSE error
LSE_error = 0
for x in range(0,data_lines):
	tmp = 0
	for y in range(0, dim):
		tmp += LSE_sol[y][0]*A[x][y]
	tmp -= data[x][0]
	LSE_error += tmp**2
for x in range(0, dim):
	LSE_error += lam*LSE_sol[x][0]**2
print("\nThe LSE error is : {}".format(LSE_error))

print("\n\n-----------------\n\n")

# Calculate Newton solution
Newton_sol = Newton(ATA, B)

# Print Newton solution
print("\nThe Newton solution is :\ny = ", end="")
for x in range(0, final_rows):
	print("{} x^{}".format(Newton_sol[x][0], x), end=" ")
	if x is not final_rows-1:
		print("+", end=" ")
print("")

# Print Newton error
Newton_error = 0
for x in range(0,data_lines):
	tmp = 0
	for y in range(0, dim):
		tmp += Newton_sol[y][0]*A[x][y]
	tmp -= data[x][0]
	Newton_error += tmp**2
print("\nThe Newton error is : {}".format(Newton_error))