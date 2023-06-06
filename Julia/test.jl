using Plots

function TimeOrderedHeisenbergTimeEvolve(H::Function, dt::Float64, tmax::Float64, groundStateInitial::ComplexF64=1.0+0.0im)::Array{ComplexF32}
    p10 = groundStateInitial+0im
    p20 = (1.0 - p10^2)
    ps = [p10;p20]
    N = Integer(fld(tmax, dt) + 1)
    data = Array{ComplexF64, 2}(undef, N, 2)
    Hprev = I
    for (i, t) in enumerate(range(0,tmax, N))
        t = convert(Float64,t)
        Hprev = Hprev*exp(H(t))
        data[i, :] = Hprev*ps
    end
    return data
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

A = [0 1; 1 0]
v = [2;1]
v3 = [2,1]
v2 = A*v3
println(v2)
println(size(v2))
println(v)
println(size(v))
println(v3)
println(size(v3))
println(v==v3)


