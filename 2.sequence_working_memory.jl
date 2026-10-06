using LinearAlgebra, Random, Statistics, MultivariateStats, Plots
include("1.functions.jl")

N = 100
ρ = 0.9
α = 1.0
T = 2000
burnin = 100
max_lag = 20


function lag_gram(N, ρ, max_lag)
    W = random_reservoir(N, ρ)
    w_in = randn(N)
    K = hcat([W^τ * w_in for τ in 1:max_lag]...)
    K ./= mapslices(norm, K, dims=1)
    return abs.(K' * K)
end

G = mean(lag_gram(N, ρ, max_lag) for _ in 1:100)
heatmap(G, xlabel="Lag index", ylabel="Lag index", title="Lag Gram Matrix (ρ=$ρ)")

W = random_reservoir(N, ρ)
w_in = randn(N)
u = randn(T + burnin)
X = simulate_esn(W, w_in, u, T + burnin, α)
X = X[:, burnin+1:end] .- mean(X[:, burnin+1:end], dims=2)
targets = lagged_targets(u[burnin+1:end], max_lag)

pca = fit(PCA, X; maxoutdim=N, pratio=1 - 1e-12);
Y_pc = projection(pca)' * X

# correlation of the kth PC with the input at lag τ
pc_lag = [cor(Y_pc[k, :], targets[τ, :]) for k in 1:max_lag, τ in 1:max_lag]
#need to run many trials to get the clear relationship but here is one example
heatmap(pc_lag, xlabel="PC index", ylabel="Lag index", title="Correlation of PCs with lagged input")
heatmap(abs.(pc_lag), xlabel="PC index", ylabel="Lag index", title="Abs.Correlation of PCs with lagged input")

# readouts
max_lag = 5
# theory: w_out(k) = W^k w_in
W_out_theory = hcat([W^k * w_in for k in 1:max_lag]...)
Y_theory = W_out_theory' * X
Y_theory ./= std(Y_theory, dims=2)

# Hebbian: leading PCs learned online
Xs = X ./ std(X)
W_hebb = learn_pcs_sanger(Xs, max_lag; η=1e-3, epochs=1000) # tweak these parameters for better convergence
Y_hebb = W_hebb * X

# the hebbian rule only finds each direction up to sign, so here we flip 
# each output to match the sign of its target lag
function match_to_lags(Y, targets, max_lag)
    Ym = similar(Y, max_lag, size(Y, 2))
    for τ in 1:max_lag
        c = [cor(Y[j, :], targets[τ, :])^2 for j in axes(Y, 1)]
        j = argmax(c)
        s = sign(cor(Y[j, :], targets[τ, :]))
        Ym[τ, :] = s * Y[j, :]
    end
    return Ym
end

Y_hebb_matched = match_to_lags(Y_hebb, targets, max_lag)
Y_hebb_matched ./= std(Y_hebb_matched, dims=2)

# example outputs: target vs theory vs Hebbian
# sometimes the hebbian readout is not very well aligned - it can depend on various factors, mainly needing convergence of learning
tw = 100:150
plot(heatmap(targets[1:max_lag, tw], title="Target", c=:balance, colorbar=false),
     heatmap(Y_theory[:, tw], title="Theory", c=:balance, colorbar=false),
     heatmap(Y_hebb_matched[:, tw], title="Hebbian", c=:balance, colorbar=false),
     layout=(1, 3), size=(1200, 400), ylabel="Lag", xlabel="Time")


# input alignment: aligning w_in with the leading right singular vector of
# W^k concentrates variance into the lag-k channel (Fig. 2D,E)
function aligned_win(W, k)
    U, S, V = svd(W^k)
    return V[:, 1]
end

# variance in the lag-k PC and recall at lag k, for a given input layer
function lag_k_stats(W, w_in, u, k, α, burnin)
    N = size(W, 1)
    X = simulate_esn(W, w_in, u, length(u), α)
    X = X[:, burnin+1:end] .- mean(X[:, burnin+1:end], dims=2)
    tgt = lagged_targets(u[burnin+1:end], k)[k, :]

    pca = fit(PCA, X; maxoutdim=min(N, k + 5), pratio=1 - 1e-12)
    Y = projection(pca)' * X
    j = argmax([cor(Y[i, :], tgt)^2 for i in axes(Y, 1)])
    v = var(Y[j, :])

    y = vec((W^k * w_in)' * X)
    return v, abs(cor(y, tgt))
end


# random vs aligned input across lags
function alignment_experiment(N, ρ, α, T, burnin, max_k; trials=20)
    var_rand = zeros(max_k)
    var_align = zeros(max_k)
    ratio = zeros(max_k)
    for k in 1:max_k
        vr = zeros(trials); va = zeros(trials); rr = zeros(trials)
        for t in 1:trials
            W = random_reservoir(N, ρ)
            u = randn(T + burnin)

            w_rand = randn(N); w_rand .*= 1.0 / norm(w_rand)
            w_align = aligned_win(W, k); w_align .*= 1.0 / norm(w_align)

            vr[t], ρ_rand = lag_k_stats(W, w_rand, u, k, α, burnin)
            va[t], ρ_align = lag_k_stats(W, w_align, u, k, α, burnin)

        end
        var_rand[k] = mean(vr)
        var_align[k] = mean(va)
    end
    return var_rand, var_align
end

max_k = 10
#need to run many trials but here is an example
var_rand, var_align = alignment_experiment(N, ρ, α, 2000, burnin, max_k; trials=20)
bar(1:max_k  .+ 0., var_align, c=:purple, lw=3, marker=:circle, label="Aligned")
bar!(1:max_k, var_rand, c=:cyan, lw=3, marker=:circle, label="Random",
      xlabel="Lag", ylabel="Variance in lag-relevant PC",
      title="Input alignment concentrates variance", grid=false)



# Fig 2D/E input alignment
function alignment_ratios(N, ρ, α, ι, T, max_lag; trials=100)
    theory = Vector{Vector{Float64}}(undef, trials)
    pcr    = Vector{Vector{Float64}}(undef, trials)
    hebb   = Vector{Vector{Float64}}(undef, trials)

    for trial in 1:trials
        W = random_reservoir(N, ρ)
        u = randn(T)
        targets = lagged_targets(u, max_lag)
        targets .-= mean(targets, dims=2)

        tr = zeros(max_lag); pr = zeros(max_lag); hr = zeros(max_lag)

        for lag in 1:max_lag
            U, S, V = svd(W^lag)
            w_opt = V[:, 1]; w_opt .*= ι
            w_rand = randn(N); w_rand .*= ι / norm(w_rand)

            X_rand = simulate_esn(W, w_rand, u, T, α); X_rand .-= mean(X_rand, dims=2)
            X_opt  = simulate_esn(W, w_opt,  u, T, α); X_opt  .-= mean(X_opt, dims=2)

            pca_rand = fit(PCA, X_rand; maxoutdim=N, pratio=1 - 1e-10)
            pca_opt  = fit(PCA, X_opt;  maxoutdim=N, pratio=1 - 1e-10)

            # theory readout W^lag w_in
            yt_rand = vec((W^lag * w_rand)' * X_rand); yt_rand ./= std(yt_rand)
            yt_opt  = vec((W^lag * w_opt)'  * X_opt);  yt_opt  ./= std(yt_opt)

            # PC readout
            yp_rand = vec(projection(pca_rand)[:, lag]' * X_rand); yp_rand ./= std(yp_rand)
            yp_opt  = vec(projection(pca_opt)[:, lag]'  * X_opt);  yp_opt  ./= std(yp_opt)

            # Hebbian readout
            yh_rand = vec(learn_pcs_sanger(X_rand, lag; η=0.1, epochs=1)[lag, :]' * X_rand); yh_rand ./= std(yh_rand)
            yh_opt  = vec(learn_pcs_sanger(X_opt,  lag; η=0.1, epochs=1)[lag, :]' * X_opt);  yh_opt  ./= std(yh_opt)

            # drop transient, sign-align against normalised target
            yt_rand = yt_rand[lag+1:end]; yt_opt = yt_opt[lag+1:end]
            yp_rand = yp_rand[lag+1:end]; yp_opt = yp_opt[lag+1:end]
            yh_rand = yh_rand[lag+1:end]; yh_opt = yh_opt[lag+1:end]

            t = (targets[lag, :] ./ std(targets[lag, :]))[1:length(yt_rand)]

            align!(y) = (cor(y, t) < 0 && (y .*= -1); y)
            align!.((yt_rand, yt_opt, yp_rand, yp_opt, yh_rand, yh_opt))

            tr[lag] = abs(cor(yt_opt, t)) / abs(cor(yt_rand, t))
            pr[lag] = abs(cor(yp_opt, t)) / abs(cor(yp_rand, t))
            hr[lag] = abs(cor(yh_opt, t)) / abs(cor(yh_rand, t))
        end

        theory[trial] = tr; pcr[trial] = pr; hebb[trial] = hr
    end
    return theory, pcr, hebb
end

theory_ratio, pc_ratio, hebb_ratio = alignment_ratios(N, ρ, α, 1., 2000, 10; trials=1000)
theory_mean = vec(mean(hcat(theory_ratio...), dims=2))

#need to run many trials but here is example with THEORY only
plot(1:10, theory_mean, c=:magenta3, lw=3, marker=:circle, label="Theory (aligned/random)")
hline!([1.0], c=:red, ls=:dash, label="No improvement",
       xlabel="Lag", ylabel="Improvement ratio",
       title="Aligned input improves recall", grid=false)

