"""
<Python for data analysis>
--------------------------
Chapter 4: NumPy Basics: Arrays and Vectorized Computation
"""


import numpy as np
from numpy.random import randn

import matplotlib.pyplot as plt

""" Creating ndarrays """
# Passing a list
data1 = [6, 7.5, 8, 0, 1]

arr1 = np.array(data1)

arr1
arr1.dtype

# Nested sequences, like a list of equal-length lists, will be converted into a multidimensional array:
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]

arr2 = np.array(data2)

arr2
arr2.ndim
arr2.shape
arr2.dtype

# Zeros and ones create arrays of 0’s or 1’s, respectively, with a given length or shape,
# pass a tuple for the shape
np.zeros(10)
np.zeros((3, 6))

# Arange is an array-valued version of the built-in Python range function:
np.arange(15)

# Create a square N x N identity matrix (1’s on the diagonal and 0’s elsewhere)
np.identity(4)
np.eye(4)

# Convert or cast an array from one dtype to another using ndarray’s astype method:
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype

# If I cast some floating point numbers to be of integer dtype, the decimal part will be truncated:
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr.astype(np.int32)

# Should you have an array of strings representing numbers, you can use astype to convert them to numeric form:
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)

# Another array’s dtype attribute:
int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)


"""Basic Indexing and Slicing"""
# An important first distinction from lists is that array slices are views on the original array. 
# This means that the data is not copied, and any modifications to the view will be reflected in 
# the source array:

l = list(range(10))
arr = np.arange(10)

l2 = l[5:8]
arr_slice = arr[5:8]

l2[1] = 12345
arr_slice[1] = 12345

l
arr

# If you want a copy of a slice of an ndarray instead of a view, you will
# need to explicitly copy the array; for example  arr[5:8].copy() 
arr_slice = arr[5:8].copy()
arr_slice[1] = 12345
arr

# Pass a comma-separated list of indices to select individual elements in higher dimensional arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]
arr2d[0][2]
arr2d[0, 2]

# In multidimensional arrays, if you omit later indices, the returned object will be a 
# lowerdimensional ndarray consisting of all the data along the higher dimensions.
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]

# Both scalar values and arrays can be assigned to arr3d[0] :
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d


"""Boolean Indexing"""
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = randn(7, 4)
names == 'Bob'

# The boolean array must be of the same length as the axis it’s indexing
data[names == 'Bob']
data[names == 'Bob', 2:]
data[names == 'Bob', 3]

# To select everything but  'Bob' , you can either use  != or negate the condition using -:
names != 'Bob'
data[names != 'Bob']
data[-(names == 'Bob')]

# Selecting two of the three names to combine multiple boolean conditions, use boolean
# arithmetic operators like  & (and) and  | (or):
mask = (names == 'Bob') | (names == 'Will')
data[mask]


"""Fancy Indexing"""
# Fancy indexing is a term adopted by NumPy to describe indexing using integer arrays.
# To select out a subset of the rows in a particular order
arr = np.empty((8, 4))

for i in range(8):
	arr[i] = i

arr[[4, 3, 0, 6]]
arr[[-3, -5, -7]]

# Passing multiple index arrays does something slightly different; it selects 
# a 1D array of elements corresponding to each tuple of indices:
arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]

# Get the rectangular region formed by selecting a subset of the matrix’s rows and columns:
arr[[1, 5, 7, 2]][[0, 3, 1, 2]]
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

# Another way is to use the  np.ix_ function, which converts two 1D integer arrays to an
# indexer that selects the square region:
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]

# Keep in mind that fancy indexing, unlike slicing, always copies the data into a new array.


"""Transposing Arrays and Swapping Axes"""
# Arrays have the  transpose method and also the special T attribute:
arr = np.arange(15).reshape((3, 5))
arr.T

# computing the inner matrix product using np.dot :
arr = randn(6, 3)
np.dot(arr.T, arr)

# For higher dimensional arrays, transpose will accept a tuple of axis numbers 
# to permute the axes (for extra mind bending):
arr = np.arange(16).reshape((2, 2, 4))
arr
arr.transpose((1, 0, 2))

# ndarray has the method  swapaxes which takes a pair of axis numbers:
# swapaxes similarly returns a view on the data without making a copy.
arr.swapaxes(1, 2)


"""Universal Functions: Fast Element-wise Array Functions"""
# Many ufuncs are simple elementwise transformations, like sqrt or exp:
arr = np.arange(10) 
np.sqrt(arr)
np.exp(arr)

#  add or maximum , take 2 arrays and return a single array as the result:
x = randn(8)
y = randn(8)
np.maximum(x, y)
np.add(x, y)

# modf Return fractional and integral parts of array as separate array:
arr = randn(7) * 5
np.modf(arr)

"""Unary ufuncs:
	------------
	abs, fabs :
		Compute the absolute value element-wise for integer, floating point, or complex values.
		Use  fabs as a faster alternative for non-complex-valued data
	sqrt :
		Compute the square root of each element. Equivalent to  arr ** 0.5
	square :
		Compute the square of each element. Equivalent to  arr ** 2
	exp :
		Compute the exponent e x of each element
	log, log10, log2, log1p :
		Natural logarithm (base e), log base 10, log base 2, and log(1 + x), respectively
	sign :
		Compute the sign of each element: 1 (positive), 0 (zero), or -1 (negative)
	ceil :
		Compute the ceiling of each element, i.e. the smallest integer greater than or equal to
		each element
	floor :
		Compute the floor of each element, i.e. the largest integer less than or equal to each
		element
	rint :
		Round elements to the nearest integer, preserving the  dtype
	modf :
		Return fractional and integral parts of array as separate array
	isnan :
		Return boolean array indicating whether each value is  NaN (Not a Number)
	isfinite, isinf :
		Return boolean array indicating whether each element is finite (non- inf , non- NaN ) or
		infinite, respectively
	cos, cosh, sin, sinh, tan, tanh :
		Regular and hyperbolic trigonometric functions
	arccos, arccosh, arcsin, arcsinh, arctan, arctanh :
		Inverse trigonometric functions
	logical_not :
		Compute truth value of  not x element-wise. Equivalent to  -arr """


"""Binary universal functions:
   ---------------------------
	add :
		Add corresponding elements in arrays
	subtract :
		Subtract elements in second array from first array
	multiply :
		Multiply array elements
	divide, floor_divide :
		Divide or floor divide (truncating the remainder)
	power :
		Raise elements in first array to powers indicated in second array
	maximum, fmax :
		Element-wise maximum.  fmax ignores  NaN
	minimum, fmin :
		Element-wise minimum.  fmin ignores  NaN
	mod :
		Element-wise modulus (remainder of division)
	copysign :
		Copy sign of values in second argument to values in first argument
	greater, greater_equal, less, less_equal, equal, not_equal :
		Perform element-wise comparison, yielding boolean array. Equivalent to infix operators
		>, >=, <, <=, ==, !=
	logical_and, logical_or, logical_xor :
		Compute element-wise truth value of logical operation. Equivalent to infix operators &
		|, ^
"""


"""Data Processing Using Arrays"""
# The  np.meshgrid function takes two 1D arrays and produces two 
# 2D matrices corresponding to all pairs of  (x, y) in the two arrays:
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)

z = np.sqrt(xs ** 2 + ys ** 2)
plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.show()


"""Expressing Conditional Logic as Array Operations"""
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

# Take a value from xarr whenever the corresponding value in
# cond is True otherwise take the value from  yarr .
result = np.where(cond, xarr, yarr)

# Suppose you had a matrix of randomly generated data
# and you wanted to replace all positive values with 2 and all negative values with -2.
# This is very easy to do with np.where :
arr = randn(4, 4)
arr
np.where(arr > 0, 2, -2)

# set only positive values to 2
np.where(arr > 0, 2, arr)

# More complicated logic
result = []
for i in range(n):
	if cond1[i] and cond2[i]:
		result.append(0)
	elif cond1[i]:
		result.append(1)
	elif cond2[i]:
		result.append(2)
	else:
		result.append(3)

# this for loop can be converted into a nested where expression:
np.where(cond1 & cond2, 0,
			np.where(cond1, 1,
						np.where(cond2, 2, 3)))


"""Mathematical and Statistical Methods"""
arr = np.random.randn(5, 4) # normally-distributed data
arr.mean()
np.mean(arr)
arr.sum()

# Functions like  mean and sum take an optional  axis argument which computes the statistic
# over the given axis, resulting in an array with one fewer dimension:
arr.mean(axis=1)
arr.sum(0)

# Other methods like  cumsum and  cumprod do not aggregate, instead producing an array
# of the intermediate results:
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

arr.cumsum(0)
arr.cumprod(1)


"""Methods for Boolean Arrays"""
# sum is often used as a means of counting True values in a boolean array:
arr = randn(100)
(arr > 0).sum() # Number of positive values

# any tests whether one or more values in an array is True , 
# while all checks if every value is True :
bools = np.array([False, False, True, False])
bools.any()
bools.all()


"""Array set operations"""
# unique(x) :Compute the sorted, unique elements in x
arr = np.array([1, 2, 2, 3, 4, 4, 5, 6])
np.unique(arr)

# intersect1d(x, y) :Compute the sorted, common elements in x and y
arr2 = np.array([2, 4, 8, 10, 13])
np.intersect1d(arr, arr2)

# union1d(x, y) :Compute the sorted union of elements
np.union1d(arr, arr2)

# in1d(x, y) :Compute a boolean array indicating whether each element of x is contained in y
np.in1d(arr, arr2)

# setdiff1d(x, y) :Set difference, elements in x that are not in y
np.setdiff1d(arr, arr2)

# setxor1d(x, y) :Set symmetric differences; elements that are in either of the arrays, but not both3
np.setxor1d(arr, arr2)


"""Storing Arrays on Disk in Binary Format"""
# np.save and np.load :saving and loading array data on disk. 
arr = np.arange(10)
# If the file path does not already end in  .npy , the extension will be appended
np.save('some_array', arr)

np.load('some_array.npy')

# Save multiple arrays in a zip archive using np.savez 
# and passing the arrays as keyword arguments:
arr2 = np.arange(10) + 10
np.savez('array_archive.npz', a=arr, b=arr2)

# Loads the individual arrays lazily when loading an .npz file:
arch = np.load('array_archive.npz')
arch['a']
arch['b']


""" linear Algebra"""
# Matrix multiplication :
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])

x.dot(y) # equivalently np.dot(x, y)
np.dot(x, np.ones(3))

# numpy.linalg has a standard set of matrix decompositions and things like inverse and determinant.
from numpy import linalg

X = randn(5, 5)
mat = X.T.dot(X)
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)



