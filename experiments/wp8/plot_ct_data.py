"""an example code from dries
    """

import numpy as np
import matplotlib.pyplot as plt


filename = 'C:/tmp/cyl8.bin'

with open(filename, mode='rb') as file:
    ct = np.fromfile(file, np.float32)
ct.shape = (760, 810, 815)
ct = np.transpose(ct, [1, 2, 0])
ct = ct[97:717, 97:717, 43:699]
ct = (ct-28000)/7000
# Show CT slice
plt.imshow(ct[:, :, 100])
plt.show()
