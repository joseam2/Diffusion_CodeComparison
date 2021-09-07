# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:17:31 2021

@author: Jose
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
import glob
import imageio

start_time = time.time()

dx = 0.01
D = 0.5
Time = 4.0
N = M = 1024
x = np.arange(-5.0, 5.24, dx)
y = np.arange(-5.0, 5.24, dx)
xx, yy = np.meshgrid(x, y)



#read data
files = glob.glob('*.txt')
# images = np.array([])
images = []

for file in files:
    filebase = file[:-4]
    print(filebase)
    psi = np.genfromtxt(file, delimiter=",")

    #Plot 3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, psi, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    surf.set_clim(-1,1)
    # Customize the z axis.
    ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=15)
    

    plt.savefig(filebase+'.png')
    # np.append(images, imageio.imread(filebase+'.png'))
    images.append(imageio.imread(filebase+'.png'))
    
imageio.mimsave('Diffusion2D_numba.gif', images)
    
    
end_time = time.time()
print(end_time-start_time)
