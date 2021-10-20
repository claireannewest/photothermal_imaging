
import numpy as np
y = np.array([10, 13, 14])
x = np.array([1, 10, 12, 14, 7, 13])
indices = np.where(np.in1d(x, y))[0]
print(x[indices])