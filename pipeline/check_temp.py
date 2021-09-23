import numpy as np
import sys



def check_if_negative(filename):
	temp_file = np.loadtxt(filename)
	x = temp_file[:,0]
	y = temp_file[:,1]
	z = temp_file[:,2]
	T = temp_file[:,3]
	return np.any(T<0)

file=str(sys.argv[1])+str("/temp.out")
print(check_if_negative(filename=file))
