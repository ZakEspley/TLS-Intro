#=
TLS:
- Julia version: 
- Author: zak
- Date: 2022-12-24
=#

using LinearAlgebra
import Dates as DT
import HDF5 as h5
using Base.Threads
using ProgressBars

Sx = 1/2 * [0 1; 
            1 0]
Sy = 1/2 * [0 -1im; 
            1im 0]
Sz = 1/2 * [1 0;
            0 -1]
Sp = 1/2 * (Sx + 1im*Sy)
Sm = 1/2 * (Sx - 1im*Sy)
I = [1 0;
     0 1]

function prefix(x, digits=nothing)
    prefixes = Dict(
        24 => "Y",
        21=> "Z",
        18=> "E",
        15=> "P",
        12=> "T",
        9=> "G",
        6=> "M",
        3=> "k",
        0=> "",
        -3=> "m",
        -6=> "µ",
        -9=> "n",
        -12=> "p",
        -15=> "f",
        -18=> "a",
        -21=> "z",
        -24=> "y",
    )
    exponent = floor(log10(x))
    roundedExp = fld(exponent,3) * 3
    # if roundedExp ∉ prefixes
    #     raise ValueError(f"The value: {x} is too large or small for a prefix.")
    # end
    prefix = prefixes[roundedExp]
    value = x / (10 ^ roundedExp)
    if digits === nothing
        return "$(value) (prefix)"
    else
        return "$(round(value;digits=digits)) $(prefix)"
    end
end

function HMaker(ω0::Float64, v::Float64, ω::Float64, Ωx::Float64, Ωz::Float64)::Function
    v = v/ω0
    ω = ω/ω0
    Ωx = Ωx/ω0
    Ωz = Ωz/ω0
    function H(t::Float64)
        return -(ω - v) * Sz - Ωx * Sx - 2 * Ωz * cos(v * t) * Sz - Ωx * (Sp * exp(-2im * v * t) + Sm * exp(2im * v * t))
    end
    return H
end

function RungeKuttaTimeEvolve(H::Function, dt::Float64, tmax::Float64, groundStateInitial::ComplexF64=1.0+0.0im)::Array{ComplexF32}
    p10 = groundStateInitial+0im
    p20 = (1.0 - p10^2)
    ps = [p10;p20]
    N = Integer(fld(tmax, dt) + 1)
    data = Array{ComplexF64, 2}(undef, N, 2)
    for (i, t) in enumerate(range(0,tmax, N))
        t = convert(Float64,t)
        h1 = H(t)
        h2 = H(t+dt/2)
        h3 = H(t+dt)

        k1 = h1*ps*-1im
        k2 = (h2 * (ps+dt*k1/2)) * -1im
        k3 = (h2 * (ps+dt*k2/2)) * -1im
        k4 = (h3 * (ps + dt*k3) ) * -1im
        ps = ps + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt
        data[i, :] = ps
    end
    return data
end

function TimeOrderedHeisenbergTimeEvolve(H::Function, dt::Float64, tmax::Float64, groundStateInitial::ComplexF64=1.0+0.0im)::Array{ComplexF32}
    p10 = groundStateInitial+0im
    p20 = (1.0 - p10^2)
    ps = [p10;p20]
    N = Integer(fld(tmax, dt) + 1)
    data = Array{ComplexF64, 2}(undef, N, 2)
    Hprev = I
    for (i, t) in enumerate(range(0,tmax, N))
        t = convert(Float64,t)
        Hprev = Hprev*exp(-1im*H(t)*dt)
        data[i, :] = Hprev*ps
    end
    return data
end

function createRun(h5File)
    today = DT.format(DT.today(), "yyyy-mm-dd")
    newgroup = true
    if today ∉ keys(h5File)
        group = h5.create_group(h5File, today)
    else
        group = h5File[today]
    end
    run = h5.create_group(group, "run"*string(length(group)+1))
    return group, run      
end

function FrequencySweepSimulation(
    ω0::Float64,
    ω::Float64,
    vmin::Float64,
    vmax::Float64,
    nFrequencies::Int64,
    dt::Float64,
    tmax::Float64,
    Ωx::Float64,
    Ωz::Float64,
    filename::String = "TLS-data.h5",
    compression::Int64 = 7,
    simulation_method::Function = RungeKuttaTimeEvolve
)
    data = h5.h5open(filename, "cw")
    frequencies = range(vmin, vmax, nFrequencies)
    group, run = createRun(data)
    h5.write_attribute(run, "dt", dt)
    h5.write_attribute(run, "tmax", tmax)
    h5.write_attribute(run, "ω0", ω0)
    h5.write_attribute(run, "ω", ω)
    h5.write_attribute(run, "Ωx", Ωx)
    h5.write_attribute(run, "Ωz", Ωz)
    h5.write_attribute(run, "v", Array(frequencies))

    dt = dt*ω0
    tmax = tmax*ω0
    @threads for v in frequencies
        H = HMaker(ω0, v, ω, Ωx, Ωz)
        probability_amplitudes = simulation_method(H, dt, tmax)
        dsName = "f=$(prefix(v))Hz-Ωx=$(prefix(Ω))Hz"
        run[dsName] = probability_amplitudes
    end
    close(data)
end

function FrequencyRabiSweepSimulation(
    ω0::Float64,
    ω::Float64,
    vmin::Float64,
    vmax::Float64,
    nFrequencies::Int64,
    dt::Float64,
    tmax::Float64,
    Ωxmin::Float64,
    Ωxmax::Float64,
    nΩx::Int64,
    Ωz::Float64;
    filename::String = "TLS-data.h5",
    compression::Int64 = 7,
    simulation_method::Function = RungeKuttaTimeEvolve
)
    data = h5.h5open(filename, "cw")
    frequencies = range(vmin, vmax, nFrequencies)
    rabiFrequencies = range(Ωxmin, Ωxmax, nΩx)
    group, run = createRun(data)
    h5.write_attribute(run, "dt", dt)
    h5.write_attribute(run, "tmax", tmax)
    h5.write_attribute(run, "ω0", ω0)
    h5.write_attribute(run, "ω", ω)
    h5.write_attribute(run, "Ωx", Array(rabiFrequencies))
    h5.write_attribute(run, "Ωz", Ωz)
    h5.write_attribute(run, "v", Array(frequencies))
    dt = dt*ω0
    tmax = tmax*ω0

    for Ωx in ProgressBar(rabiFrequencies)
        rabiSet = h5.create_group(run, "Ωx=$(prefix(Ωx,4))")
        h5.write_attribute(rabiSet, "Ωx", Ωx)
        @threads for v in frequencies
            H = HMaker(ω0, v, ω, Ωx, Ωz)
            probability_amplitudes  = simulation_method(H, dt, tmax)
            dsName = "f=$(prefix(v,4))Hz-Ωx=$(prefix(Ωx,4))Hz"
            rabiSet[dsName] = probability_amplitudes
            h5.write_attribute(rabiSet[dsName], "Ωx", Ωx)
            h5.write_attribute(rabiSet[dsName], "v", v)
        end
    end
    close(data)
end



# Default Frequency in Hz
ω0 = 1e9
#Resonant Frequency in Hz
ω = 5e9
#Time step in seconds
dt = 50e-12
#Time to run for in seconds
tmax = 600e-9
# Rabi Frequency in Hz
# Ωx = 63e6
Ωxmin = 10e6
Ωxmax = 250e6
nΩx = 25
#AC Stark Shift Frquency in Hz
Ωz = 0.0
# Min Frequency in Hz
vmin = 4.8e9
# Max Frequency in Hz
vmax = 5.2e9
# Number of frequencies
nFrequencies = 7

# FrequencySweepSimulation(
#     ω0,
#     ω,
#     vmin,
#     vmax,
#     nFrequencies,
#     dt,
#     tmax,
#     Ωx,
#     Ωz
# )

FrequencyRabiSweepSimulation(
    ω0,
    ω,
    vmin,
    vmax,
    nFrequencies,
    dt,
    tmax,
    Ωxmin,
    Ωxmax,
    nΩx,
    Ωz;
    simulation_method=TimeOrderedHeisenbergTimeEvolve
)
