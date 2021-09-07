# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:33:31 2021

@author: Jose
"""

#Diffusion Python Code

import numpy as np
from numba import jit, njit, prange, set_num_threads, get_num_threads
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

def initial():
    psi = np.zeros((N, M))

    r = np.sqrt(xx**2 + yy**2)
    psi = np.sin(r)

    return psi

# @njit(nogil=True, parallel=True)
# def Compute_P(Pcurr, Pnext, dt, iterate_arr):
@njit(nogil=True, parallel=True)
def Compute_P(Pcurr, Pnext, dt):
    #position loop
    for i in prange(1, N-1, 1):
        for j in prange(1, M-1, 1):
            Pnext[i, j] = Pcurr[i, j]+D*dt*((Pcurr[i+1, j]+Pcurr[i-1, j]+Pcurr[i, j+1]+Pcurr[i, j-1]-4.0*Pcurr[i,j])/(dx*dx))

    #BC no-flux
    Pnext[0,:] = Pnext[1,:]
    Pnext[-1,:] = Pnext[-2,:]
    Pnext[:,0] = Pnext[:,1]
    Pnext[:,-1] = Pnext[:,-2]


    return Pnext

def savepsi(psi, time):
    # return
    timetoint = np.int(time*10**4)
    timeval = "{:05d}".format(timetoint)
    np.savez_compressed('Diffusion2D_numba_{}'.format(timeval), psi)



def main_loop(totalTime):
    Kdt = 0.5
    dt = (Kdt*dx*dx/4.0)/D
    Tsave = np.arange(0, totalTime, dt*9200)
    # Tsave = np.arange(0, totalTime, dt*625)


    Psicurr = initial()
    Psinext = np.zeros((N, M))
    Psibuff = np.zeros((N, M))

    savepsi(Psicurr, 0)

    for t in np.arange(0, totalTime, dt):
        #dPsi/dt

        if(np.isin(t, Tsave)):
            savepsi(Psicurr, t)
            print('Current Time: {}'.format(t))

        set_num_threads(12)
        Psibuff = Compute_P(Psicurr, Psinext, dt)
        Psinext = Psicurr
        Psicurr = Psibuff

    # return Psicurr
    return
    
main_loop(Time)
print("Using {} threads".format(get_num_threads()))

end_time = time.time()
print(end_time-start_time)
