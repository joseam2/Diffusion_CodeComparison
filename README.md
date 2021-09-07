# Diffusion_CodeComparison
A small diffusion code that compares gpu and cpu parallelized python and julia codes

--------
Testing Hardware:

CPU: Intel Xeon 6248R (Cascade Lake), 3.0GHz, 24-core (48 threads)

GPU: Nvidia A100, 40GB

RAM: 384 GB DDR4, 3200 Mhz

--------
Time comparison:

Numba Prange code -   65.2 seconds

Numba Cuda code -     23.1 seconds

Julia Cuda code -     6.26 seconds
