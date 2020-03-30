import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('temp.out')
x_plane = -11
z_plane = -20
data_sortx = np.where(data[:,0] == x_plane)
data_sort = data[data_sortx]


# plt.scatter(data_sort[:,1], data_sort[:,2], c=data_sort[:,3])
# plt.show()

data_sortz = np.where(data_sort[:,2] == z_plane)
data_sort_new = data_sort[data_sortz]

plt.plot(data_sort_new[:,1], data_sort_new[:,3],'o')
print(max( data_sort_new[:,3]))
plt.show()