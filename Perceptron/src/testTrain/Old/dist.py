#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 500; w1 = -1; w2 = 1;
x = []; y=[];
f = open('data.txt');
for i in range(SAMPLES):
    l = f.readline();
    l.strip();
    pair = l.split();
    x.append(float(pair[0]));
    y.append(float(pair[1]));
x = np.array(x);
y = np.array(y);
"""
x = np.random.uniform(-50, 50, SAMPLES); xb=[]; xr=[];
y = np.random.uniform(-50, 50, SAMPLES); yb=[]; yr=[];
for i in range(SAMPLES):
    if (x[i]*w1+y[i]*w2>0):
        xb.append(x[i]); yb.append(y[i]);

    else:
        xr.append
"""
x1 = np.array([-50,50]);
y1 = np.array([-120,120]);

plt.xticks([-50,-25,0,25,50]);
plt.yticks([-120,-60,0,60,120]);

plt.ylim(-120.0,120.0);
plt.xlim(-50.0,50.0);
plt.plot(x,y, 'o', alpha=0.5, color='blue');
plt.plot(x1,y1, color='red');
plt.show();

