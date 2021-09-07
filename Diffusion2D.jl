#Cuda diffusion benchmark code in Julia
#Author: Jose Mancias

using CUDA, DelimitedFiles

dx = 0.01
D = 0.5
Time = 4.0
N = 1024
M = 1024
x = -5.0:dx:5.24
y = -5.0:dx:5.24

#gpu
nthreads = 32
nblocksx = Int(N / nthreads)
nblocksy = Int(M / nthreads)

function initial(psi::CuDeviceMatrix{Float32, 1}, dx_dev::Float64)
    idx_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if idx_i <= 1024 && idx_j <= 1024
        idx_x = idx_i*dx_dev - 5.0
        idx_y = idx_j*dx_dev - 5.0

        @inbounds psi[idx_i, idx_j] = sin(sqrt(idx_x*idx_x + idx_y*idx_y))
    end
    return
end

function Compute_P(Pcurr::CuDeviceMatrix{Float32, 1}, Pnext::CuDeviceMatrix{Float32, 1}, dx_dev::Float64, dt_dev::Float64, Nl::Int64, Ml::Int64)
    idx_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idx_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if idx_i < Nl && idx_j < Ml && idx_i > 1 && idx_j > 1
        @inbounds Pnext[idx_i, idx_j] = Pcurr[idx_i, idx_j] + D*dt_dev*((Pcurr[idx_i+1, idx_j]+Pcurr[idx_i-1, idx_j]+Pcurr[idx_i, idx_j+1]+Pcurr[idx_i, idx_j-1]-4.0*Pcurr[idx_i, idx_j])/(dx_dev*dx_dev))
    end

    if idx_i == 2
        Pnext[1, idx_j] = Pnext[2, idx_j]
    elseif idx_i == (Nl-1)
        Pnext[Nl, idx_j] = Pnext[Nl-1, idx_j]
    end

    if idx_j == 2
        Pnext[idx_i, 1] = Pnext[idx_i, 2]
    elseif idx_j == (Ml-1)
        Pnext[idx_i, Ml] = Pnext[idx_i, Ml-1]
    end

    return
end

function savepsi(psi::Array{Float32,2}, time::Float64)
    timetoint = round(Int64, time*10^4)
    timeval = lpad(string(timetoint), 5, '0')
    filename = string("Diff2D_julia_", timeval,".txt")
    writedlm(filename,psi,',')
    return
end


function main_loop(Time_f)
    Kdt = 0.5
    dt = (Kdt*dx*dx/4.0)/D
    Tsave = round.(collect(0.0:(dt*9200):Time_f), digits=4)
    psicurr = CUDA.fill(0.0f0, (N,M))
    @cuda blocks=(nblocksx, nblocksy) threads=(nthreads, nthreads) initial(psicurr, dx)

    psibuff = CUDA.fill(0.0f0, (N,M))
    psinext = CUDA.fill(0.0f0, (N,M))

    psicurr_h = fill(0.0f0, (N,M))

    for t = 0.0:dt:Time
        if(t in Tsave)
            copyto!(psicurr_h, psicurr)
            savepsi(psicurr_h, t)
            println(string("Current Time: ", t))
        end

	@cuda blocks=(nblocksx, nblocksy) threads=(nthreads, nthreads) Compute_P(psicurr, psinext, dx, dt, N, M)
        psibuff = psinext
        psinext = psicurr
        psicurr = psibuff


    end
return
end

#compile the code so that the code can be properly timed
@time main_loop(0.01)

#run the real code
@time main_loop(Time)
