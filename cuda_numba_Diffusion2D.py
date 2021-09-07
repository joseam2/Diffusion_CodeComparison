# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:33:31 2021

@author: Jose
"""

#Diffusion Python Code

import numpy as np
from numba import jit, njit, cuda, void, float64, int32
import time

start_time = time.time()

#Simulation
dx = 0.01
D = 0.5
Time = 4.0
N = M = 1024
x = np.arange(-5.0, 5.24, dx)
y = np.arange(-5.0, 5.24, dx)
xx, yy = np.meshgrid(x, y)


#gpu
nthreads = 32
nblocksy = N // nthreads
nblocksx = M // nthreads

def initial():
    psi = np.zeros((N, M), dtype=np.float64)
    r = np.sqrt(xx**2 + yy**2)
    psi = np.sin(r)

    return psi


@cuda.jit
def Compute_P(Pcurr, Pnext, Paramfloats, Paramints):
    dx_dev = Paramfloats[0]
    dt_dev = Paramfloats[1]
    D = Paramfloats[2]
    Nl = Paramints[0]
    Ml = Paramints[1]
    startX, startY = cuda.grid(2) #x = blockIdx.x * blockDim.x + threadIdx.x
    if startX < (Nl-1) and startY < (Ml-1) and startX >= 1 and startY >= 1:
        Pnext[startX, startY] = Pcurr[startX, startY]+D*dt_dev*((Pcurr[startX+1, startY]+Pcurr[startX-1, startY]+Pcurr[startX, startY+1]+Pcurr[startX, startY-1]-4.0*Pcurr[startX,startY])/(dx_dev*dx_dev))

    #Boundary conditions No-flux x coords
    if(startX == 1):
        Pnext[0, startY] = Pnext[1, startY]
    elif(startX == (Nl-2)):
        Pnext[-1, startY] = Pnext[-2, startY]

    #Boundary conditions No-flux y coords
    if(startY == 1):
        Pnext[startX, 0] = Pnext[startX, 1]
    elif(startY == (Ml-2)):
        Pnext[startX, -1] = Pnext[startX, -2]

    return

   
def savepsi(psi, time):
    timetoint = np.int(time*10**4)
    timeval = "{:05d}".format(timetoint)
    np.savez_compressed('Diffusion2D_numba_{}'.format(timeval), psi)

def main_loop(totalTime):
    Kdt = 0.5
    dt = (Kdt*dx*dx/4.0)/D
    Tsave = np.arange(0, totalTime, dt*9200)


    Psicurr_h = initial()
    Psinext_h = np.zeros((N, M), dtype=np.float64)
    Psibuff_h = np.zeros((N, M), dtype=np.float64)
    paramfloats_h = np.array([dx, dt , D], dtype=np.float64)
    paramints_h = np.array([N, M], dtype=np.int32)

    Psicurr = cuda.to_device(Psicurr_h)
    Psinext = cuda.to_device(Psinext_h)
    Psibuff = cuda.to_device(Psibuff_h)
    Paramfloats = cuda.to_device(paramfloats_h)
    Paramints = cuda.to_device(paramints_h)


    savepsi(Psicurr_h, 0)

    for t in np.arange(0, totalTime, dt):
        #dPsi/dt

        if(np.isin(t, Tsave)):
            Psicurr_h = Psicurr.copy_to_host()
            savepsi(Psicurr_h, t)
            print('Current Time: {}'.format(t))

        Compute_P[(nblocksx, nblocksy), (nthreads, nthreads)](Psicurr, Psinext, Paramfloats, Paramints)

        Psibuff = Psinext
        Psinext = Psicurr
        Psicurr = Psibuff

    return

main_loop(Time)


end_time = time.time()
print(end_time-start_time)
