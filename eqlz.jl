using SignalAnalysis
using Plots
using DSP
using UnderwaterAcoustics
using ToeplitzMatrices
using LinearAlgebra

plotlyjs()
fs = 44100.0
s = Float64[]
bits = Int[]
bits1 = Int[]
bb = repeat(gmseq(8), 5)
pb = real.(samples(upconvert(bb, 128, 5000.0, rcosfir(0.25, 128); fs=fs/128)))
pb = pb ./ maximum(abs.(pb)) * db2amp(180)
fc = 5000.0

function MMSE_eqlz(s, MSEQ, M)
  ϕff = xcorr(samples(s), samples(s))
  ϕfd = xcorr(MSEQ, samples(s))
  N = length(MSEQ)
  𝑅ff = ϕff[N:N+M-1]
  𝑅 = Toeplitz(𝑅ff, vec(𝑅ff))
  P = ϕfd[N:N+M-1]
  h = inv(collect(𝑅)) * (P)
  h
end

function RGN(speclvl)
  σ = db2amp(121.0 + speclvl - 68.0) * (fc/5000.0) /  √( fs/ 48000 )
  RedGaussianNoise(σ)
end

N_ff = 10
N_fb = 3
Δ = 0.01
λ = 1/0.99


function RLSLinEqlz(s, bb, N_ff)
    P = Δ * I(N_ff)
    e = Complex{Float64}[]
    d = zero(Complex{Float64})
    w = Array{Complex{Float64}}(undef, N_ff)
    eqlz = Complex{Float64}[]

    for i  ∈ 1:length(bb)-(N_ff)
      y = s[i + N_ff-1:-1:i]
      d = w' * y
      push!(eqlz,  d)
      err = bb[i] - d
      nume = collect(P * y)
      denom = λ +  y'*nume
      K = nume/denom
      PPrime = K*y'*P
      PPrime = collect(PPrime)
      P=(P-PPrime)/λ
      w = w + K*(err)
    end
    eqlz
end


function adaptiveRLS(s, bb)
    P = Δ * I(N_fb+N_ff)
    e = Complex{Float64}[]
    eqlz = Complex{Float64}[]
    w = Array{Complex{Float64}}(undef, N_ff+N_fb)

    for i ∈ 1:N_fb
      push!(eqlz, zero(Complex{Float64}))
    end

    for i  ∈ N_fb+1:length(bb)-(N_ff+N_fb)
        w_ff = w[1:N_ff]
        w_fb = w[N_ff+1:end]
        y = s[i + N_ff-1:-1:i]
        pn = w_ff' * y
        y_b = bb[i-1:-1:i-N_fb]
        qn = w_fb' * y_b
        u = [y;y_b]
        d = pn + qn
        push!(eqlz,  d)
        push!(e, bb[i] - d)
        nume = P*u
        denom = λ+ u'*nume
        K = nume/denom
        PPrime = K*u'*P
        PPrime = collect(PPrime)
        P=(P-PPrime)/λ
        w = w + K*(bb[i] - d)
    end
    eqlz, e
end

env = UnderwaterEnvironment(
  bathymetry = ConstantDepth(20.0),
  seasurface = SeaState2,
  seabed = SandyClay,
  noise = RGN(100)
)

pm = PekerisRayModel(env, 100)
arr = arrivals(pm, AcousticSource(0.0, -10.0, 5000.0), AcousticReceiver(1300.0, -5.0))
ir = impulseresponse(arr, fs; reltime=true, approx=true)
ir = ir/maximum(abs.(ir))

rx_sig = conv(ir, pb) 
rx_sig .+= record(noise(env),length(rx_sig)/fs, fs)

bbd = downconvert(rx_sig, 128, 5000.0, rcosfir(0.25, 128); fs = fs)
pad = 11
bbd = bbd[1+pad:end-pad]
bbd = bbd./maximum(abs.(bbd))

h = MMSE_eqlz(bbd, bb, 3)

bbd1=sfilt(h, 1, bbd)

(eqlz, e) = adaptiveRLS(bbd,bb)
decisions = sign.(-imag.(eqlz))

[push!(bits, sign(imag(bb[i]))==sign(imag(bbd1[i])) ? 1 : 0)  for i ∈ eachindex(bb)] 
[push!(bits1, sign(imag(bb[i]))==sign(decisions[i]) ? 1 : 0)  for i ∈ eachindex(decisions)] 
ber=sum(bits)/length(bits) 
ber1=sum(bits1)/length(bits1)

plot(imag.(bb))
plot!(imag.(samples(bbd1)))
plot!((decisions))

(eqlz)=RLSLinEqlz(bbd, bb, 2)