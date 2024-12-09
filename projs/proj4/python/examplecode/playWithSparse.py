import numpy as np
from scipy.sparse import coo_array


# Define a small (in case you want to look at it) matrix of random numbers
rng = np.random.default_rng()
mtxRandom =  rng.random((4,5))
print("Original Matrix")
print(mtxRandom)


# Set any values below a threshold to zero
mtxZeroed = mtxRandom.copy();
mtxZeroed[mtxZeroed<0.3] = 0
print("Matrix with Zeros")
print(mtxZeroed)

# Now compress the matrix using the coo_array method
mtxSparse = coo_array(mtxZeroed)
print("Sparse Array")
print(mtxSparse)