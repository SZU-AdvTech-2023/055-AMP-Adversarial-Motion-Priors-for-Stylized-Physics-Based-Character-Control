import numpy as np
from visualization.utils import quick_visualize

p_index = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
obj = np.load(r"X:\MLD\Example_100_batch0_1.npy")
print(obj.shape)
obj = np.swapaxes(obj, 0, 2)

quick_visualize(p_index, obj)

