# -*- coding: utf-8 -*-

import numpy as np
from scipy import *
from pylab import *

# Definition of funcion
x=np.arange(0,2*pi,0.1)
y1=sin(x)
y2=cos(x)

# Display graph
plot(x,y1,'b-+',x,y2,'g-o')
xlabel('x')
ylabel('y')
title('Waveforms')
grid(True)
legend(('sin(x)','cos(x)'));

# Refresh
show()

