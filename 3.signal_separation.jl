using LinearAlgebra, Random, Statistics, Distributions, Plots, MatrixEquations
using Plots.PlotMeasures
include("1.functions.jl")

dprime2(Δμ, Σ) = dot(Δμ, Σ \ Δμ)

# rank-1 (dominant mode) gain
gain_rank1(ρ, α) = (2 - α + α*ρ) / (α*(1 - ρ))

# full-spectrum: dominant mode plus bulk average over the circular law
gain_full(N, ρ, α) = gain_rank1(ρ, α) + (N - 1)*(2 - α)/α

# empirical spectral gain Σ Re[(1+λ)/(1-λ)] over eigenvalues of Weff
function gain_empirical(N, ρ, α; trials=100)
    vals = zeros(trials)
    for t in 1:trials
        W = random_reservoir(N, ρ)
        ev = eigvals(Matrix(W_eff(W, α)))
        g = real.((1 .+ ev) ./ (1 .- ev))
        vals[t] = sum(filter(isfinite, g))
    end
    return mean(vals)
end

# drive network with two mean-shifted noisy inputs, measure d'^2 in each space
function simulate_dprime(W, w_in, Δμ, Σin, α, T; burn=0.5)
    N = size(W, 1)
    dist = MvNormal(Δμ, Σin)
    b = trunc(Int, burn*T)
    xs = zeros(N, T - b)
    x = zeros(N)
    k = 1
    for t in 1:T
        x = (1-α)*x + α*(W*x + w_in*rand(dist))
        if t > b
            xs[:, k] = x; k += 1
        end
    end
    return vec(mean(xs, dims=2)), cov(xs'; corrected=false), xs
end

#empirical vs theory across input separation
N = 100
Nin = 2
deltas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]
pairs = [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1)]   # (ρ, α)
Tfactor = 50 # ensuring enough time of input to bring activity to stationarity

function sweep(pairs, deltas, trials)
    din  = zeros(length(pairs), length(deltas))
    dnet = zeros(length(pairs), length(deltas))
    for (ip, (ρ, α)) in enumerate(pairs)
        τeff = 1 / (α*(1-ρ))
        T = trunc(Int, Tfactor*τeff)
        for (id, δ) in enumerate(deltas)
            dvi = zeros(trials); dvn = zeros(trials)
            for t in 1:trials   
                W = random_reservoir(N, ρ)
                w_in = randn(N, Nin) / sqrt(N)
                v = randn(Nin); Δμ = δ * v/norm(v)
                A = randn(Nin, Nin); Σin = A*A'; Σin ./= tr(Σin)
                _, Σx, _ = simulate_dprime(W, w_in, Δμ, Σin, α, T)
                Δx = fixed_point(W, w_in, Δμ, α)
                dvi[t] = dprime2(Δμ, Σin)
                dvn[t] = dprime2(Δx, Σx)
            end
            din[ip, id]  = mean(dvi)
            dnet[ip, id] = mean(dvn)
        end
    end
    return din, dnet
end

# two-alternative decision: Fisher vs Hebbian readout
# Fisher discriminant w_out = (Σ1+Σ2)^{-1} Δx*, compared to the leading PC.
function decision_readouts(N, ρ, α, μ, T; trials=10)
    accF = zeros(trials); accH = zeros(trials)
    for tr in 1:trials
        W = random_reservoir(N, ρ)
        w_in = randn(N) / sqrt(N)
        in1 = μ .+ randn(T); in2 = -μ .+ randn(T)
        X1 = simulate_esn(W, w_in, in1, T, α)
        X2 = simulate_esn(W, w_in, in2, T, α)
        Σ1 = cov(X1'); Σ2 = cov(X2') # this is the empirical covariance
        # Σ1 = state_covariance(W, w_in, 1.0, α); Σ2 = state_covariance(W, w_in, 1.0, α) # this is the theoretical covariance
        Δx = vec(mean(X1, dims=2) - mean(X2, dims=2)) # this is the empirical mean difference
        # Δx = fixed_point(W, w_in, μ, α) # this is the theoretical fixed point difference
        wF = (Σ1 + Σ2) \ Δx
        X = hcat(X1, X2) .- mean(hcat(X1, X2), dims=2)
        wH = learn_pcs_sanger(X, 1; η=1e-3, epochs=10)'
        lab = vcat(ones(T), -ones(T))
        for (w, acc) in ((wF, accF), (wH, accH))
            y = w' * hcat(X1, X2) |> vec
            y .*= sign(dot(y, lab))
            acc[tr] = mean(sign.(y) .== lab)
        end
    end
    return mean(accF), mean(accH)
end



# gain scaling factors over (ρ, α)
ρs = 0.01:0.01:0.999 # can tweak this to bring spectral radius even closer to 1
αs = 0.001:0.01:1.0# can tweak this to bring leak rate even closer to 0
G1 = [gain_rank1(ρ, α) for α in αs, ρ in ρs]
Gf = [gain_full(N, ρ, α) for α in αs, ρ in ρs]

h1 = heatmap(ρs, αs, log10.(G1), title="Rank-1", c=:viridis)
contour!(
    ρs, αs, log10.(G1),
    linewidth=1,
    linecolor=:white,  xlabel = "Spectral radius ρ",
    ylabel = "Leak rate α",title = "Rank-1",   
    alpha=0.3, clabels=true, clim=(0,maximum(log10.(G1))),
    levels=collect(0:0.5:6)                
)
h2 = heatmap(ρs, αs, log10.(Gf), title="Full-spectrum", c=:viridis)
contour!(
    ρs, αs, log10.(Gf),
    linewidth=1,
    linecolor=:white,  xlabel = "Spectral radius ρ",
    ylabel = "Leak rate α",title = "Rank-1",   
    alpha=0.3, clabels=true, clim=(0,maximum(log10.(Gf))),
    levels=collect(0:0.5:6)                
)

plot(h1,h2,
     layout=(1, 2), xlabel="Spectral radius ρ", ylabel="Leak rate α",
     size=(900, 380), colorbar_title="log₁₀ gain", margin=5mm)


#empirical d'^2 vs input separation, three (ρ,α) regimes
din, dnet = sweep(pairs, deltas, 1000)

plts = []
for (ip, (ρ, α)) in enumerate(pairs)
    p = plot(deltas, din[ip, :], c=:grey, lw=3, label=ip == 1 ? "Input" : "",
             title="ρ=$ρ, α=$α", xlabel="‖Δμ‖", yscale=:log10, ylim=(0.01, 10^10),
             ylabel=ip == 1 ? "d'² (log)" : "", grid=false)
    plot!(p, deltas, dnet[ip, :], c=:black, lw=3, label=ip == 1 ? "RNN" : "")
    plot!(p, deltas, [gain_rank1(ρ, α) for _ in deltas] .* din[ip, :],
          c=:dodgerblue, ls=:dash, lw=2, label=ip == 1 ? "Rank-1" : "")
    plot!(p, deltas, [gain_full(N, ρ, α) for _ in deltas] .* din[ip, :],
          c=:navy, ls=:dash, lw=2, label=ip == 1 ? "Full-spectrum" : "")
    push!(plts, p)
end
plot(plts..., layout=(1, 3), size=(1300, 400), margin=10mm)


#full-spectrum approximation vs empirical, across N
Ns = round.(Int, exp10.(range(log10(20), log10(500), length=10)))
gain_pairs = [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1)]

plts = []
for (ρ, α) in gain_pairs
    emp = [gain_empirical(n, ρ, α; trials=200) for n in Ns]
    thy = [gain_full(n, ρ, α) for n in Ns]
    p = scatter(Ns, emp, c=:orange, label="Empirical", yscale=:log10,
                title="ρ=$ρ, α=$α", xlabel="N", ylabel="Spectral gain", grid=false)
    plot!(p, Ns, thy, c=:dodgerblue, lw=3, label="Theory", ylim=(20,15000))
    push!(plts, p)
end
plot(plts..., layout=(1, 3), size=(1300, 400),legend=:topleft)

#decision accuracy, Fisher vs Hebbian
μs = 0.0:0.1:1.0
plts = []
for (ρ, α) in pairs
    τeff = 1 / (α*(1-ρ))
    T = trunc(Int, Tfactor*τeff)
    accF = zeros(length(μs)); accH = zeros(length(μs))
    for (i, μ) in enumerate(μs)
        accF[i], accH[i] = decision_readouts(N, ρ, α, μ, T; trials=10)
    end
    p = plot(μs, accF, c=:magenta3, lw=3, marker=:circle, label="Fisher",
             title="ρ=$ρ, α=$α", xlabel="‖Δμ‖", ylabel="Accuracy",
             ylim=(0.45, 1.02), grid=false)
    plot!(p, μs, accH, c=:green, lw=3, marker=:circle, label="Hebbian")
    hline!(p, [0.5], c=:black, ls=:dash, label="Chance")
    push!(plts, p)
end
plot(plts..., layout=(1, 3), size=(1300, 400))