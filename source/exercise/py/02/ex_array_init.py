# -*- coding: utf-8 -*-

# Required for using NumPy
import numpy as np

# Identify each element specifically
a = np.array([0, 1, 2])
b = np.array([[0,1],[2,3]])

print a
print b

# Consecutive data
c = np.arange(10)
d = np.linspace(0,1,6)

print c
print d

# Predefined data
e = np.ones((3,2))
f = np.zeros((3,2))
g = np.eye((5))

print e
print f
print g

# Diagonal matrix
h = np.diag(np.arange(5))

print h

# Random matrix
i = np.random.rand(3,6)

print i

