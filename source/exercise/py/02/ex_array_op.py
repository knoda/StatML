# -*- coding: utf-8 -*-

import numpy as np

a = np.arange(25)

# Print all
print a

### Specify by indices

# Index order is row->column

# 4th element
print a[3]

### Slice

# Format of slice is start:end:stop

# From 3rd to 21th with skip 5
print a[2:20:5]
# Abbreviated indices mean 1st, last and step 1
print a[::5]

### Slice and view

# Slice operation creates a view.
# View is a method for accessing an array.
# Therefore, original array is not copied but a reference is returned.
# If view is modified, the original array is modified as well.

# Return with view
b = a[::2]
print b

# If b is modified, a is modified as well.
b[0] = 12
print b
print a

### Shape modification

# A vector with shape (25,1)
c = np.arange(25)
print a.shape

# Modify the shape to (5,5)
d = c.reshape((5,5))    # Return with view
print d

# If d is modified, c is modified as well.
d[0,0] = 33;
print c

### Repetition of array

# Initialization of a vector and reshape at a same time
e = np.arange(4).reshape((2,2))
print e

# Repeat array 2 times for row direction and 3 times for column direction
f = np.tile(e,(2,3))
print f

