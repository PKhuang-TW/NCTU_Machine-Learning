
# # Testing list input and transpose
# a = []

# a.append([1,2])
# a.append([3,4])
# a.append([5,6])
# a.append([7,8])
# print("1 - {}".format(a))


# a = zip(*a)
# print("2 - {}".format(a))


# a = map(list, a)
# print("3 - {}".format(a))

# print("4 - {}".format(a[1]))

# -------------------------------------------------- #

# #  Testing read file
# file = open("data.txt") 
# count = 1
# while 1:
# 	lines = file.readlines(100000)
# 	if not lines:
# 		break
# 	for line in lines:
# 		print("line.{} : {}".format(count, line[1])) # do something
# 		count += 1
# file.close()

# -------------------------------------------------- #

# Testing Matrix Mult.
# List1 = [1,2,3,4]
# List2 = [5,6,7,8]
# List3 = map(lambda (a,b):a*b,zip(List1,List2))
# print List3

# -------------------------------------------------- #

A = [[4,1],[5,2],[6,3]]

count_x = 0
for x in A:
	count_y = 0
	for y in x:
		A[count_x][count_y] = int(y*2)
		count_y += 1
	count_x += 1
print A