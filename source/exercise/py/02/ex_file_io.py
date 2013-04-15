# -*- coding: utf-8 -*-

import os, os.path
import numpy as np
import scipy
import scipy.misc
from scipy import *
from pylab import *

### Directory operation

# Get current directory
current_dir = os.getcwd()
print current_dir

# Edit path
data_dir = os.path.join(current_dir, 'data')
print data_dir

# Check directory existence and creation
if not (os.path.exists(data_dir)):
    os.mkdir('data')
    print "data dir created"

# Directory change
os.chdir(data_dir)

### Save and load

a = np.arange(10).reshape((2,5))

# Save with txt format
np.savetxt('integers.txt', a)

# Save with binary format
np.save('integers.npy', a)

# Load from txt format
b = np.loadtxt('integers.txt')

# Load from binary format
c = np.load('integers.npy')

# Load from MATLAB format
#scipy.io.loadmat()

# Lena-chan
lena = scipy.misc.lena()

# Save image
imsave('lena.png', lena, cmap=cm.gray)

# Load image
lena_reloaded = imread('lena.png')

# Disp image
imshow(lena_reloaded, cmap=cm.gray)

# Save figure
savefig('lena_fig.png')

# Show window
show()

