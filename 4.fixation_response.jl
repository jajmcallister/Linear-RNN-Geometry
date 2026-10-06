using LinearAlgebra, Random, Statistics
include("1.functions.jl")

N = 100
ρ = 0.9
α = 1.0


function run_fixate_response_readouts(N, α, rho, T_input, T_relax)

    W = randn(N, N) / sqrt(N)          # recurrent weights
    W .*= rho / maximum(abs.(eigvals(W)))  # scale to desired spectral radius
    Win0 = randn(N)                          # initial input vector
    W_eff = (1-α)*I + α*W
    v1 = eigen(W_eff).vectors[:,1]               # leading eigenvector
    v1 = real.(v1)        
    v1 ./= norm(v1)
    μ₁ = 1.0; μ₂ = -1.0 # two different stimuli means
    input_std = 0.1 # std of input noise

    # #option1: just one trial of each stimulus
    # u_train = vcat(μ₁ .+ randn(trunc(Int,T_input)) * input_std, zeros(T_relax),
    #                 μ₂ .+ randn(trunc(Int,T_input)) * input_std, zeros(T_relax),
    #                 μ₁ .+ randn(trunc(Int,T_tail)) * input_std)
    #option2: many trials of each stimulus 
    u_train = vcat([
                vcat(μ .+ randn(T_input) * input_std, zeros(T_relax))
                for μ in repeat([μ₁, μ₂], 5)
            ]...)


    u_test = vcat(μ₁ .+ randn(trunc(Int,T_input)) * input_std, zeros(T_relax),
                    μ₂ .+ randn(trunc(Int,T_input)) * input_std, zeros(T_relax))

    y_test = vcat(zeros(T_input), ones(T_relax), zeros(T_input), -ones(T_relax))

    X_train = simulate_esn_ddm(
        W, Win0, u_train, length(u_train), α,
        T_input, T_relax
    )

    X_test = simulate_esn_ddm(
        W, Win0, u_test, length(u_test), α,
        T_input, T_relax
    )

    # Theory-based 1
    A = inv(I-W_eff)
    Win_theory = Win0 - (v1' * A * Win0) / norm(A'*v1)^2 * (A'*v1)
    X_test_theory = simulate_esn_ddm(
        W, Win_theory, u_test, length(u_test), α,
        T_input, T_relax
    )
    W_out_theory = real.(v1)

    # Data-driven Rayleigh / GEP - theory II
    T_train = length(u_train)
    T_fix1 = 1:T_input
    T_fix2 = (T_input+T_relax+1):(2*T_input+T_relax)
    T_fix = vcat(T_fix1, T_fix2)

    # Response periods: the original relax windows, but NOT the final tail
    # (tail is excluded from Cov_fix/Cov_resp fitting, kept only for the plotting)
    T_resp1 = (T_input+1):(T_input+T_relax)
    T_resp2 = (2*T_input+T_relax+1):(2*T_input+2*T_relax)
    T_resp = vcat(T_resp1, T_resp2)

    Xc = X_train .- mean(X_train, dims=2)
    X_test_c = X_test .- mean(X_train, dims=2)

    Cov_fix  = cov(Xc[:,T_fix]')
    Cov_resp = cov(Xc[:,T_resp]')

    γ = 1e-6
    λ, W_gep = eigen(Cov_resp, Cov_fix + γ*I)
    idx_max = argmax(abs.(λ))
    W_out_data = real.(W_gep[:, idx_max])

    #Sanger's rule / Hebbian
    Xs = Xc ./ std(Xc)
    W_fix = learn_pcs_sanger(Xs[:,T_fix], 5; η=1e-3, epochs=1000)
    P_fix = W_fix' * W_fix
    X_ortho = Xs[:,T_resp] .- P_fix * Xs[:,T_resp]
    W_resp = learn_pcs_sanger(X_ortho, 1; η=1e-3, epochs=1000)
    W_out_sanger = real.(W_resp[1,:]); W_out_sanger ./= norm(W_out_sanger)
    
    W_out_theory = real.(v1)                                 # already unit norm, N-vector
    W_out_data = real.(W_gep[:, idx_max]); W_out_data ./= norm(W_out_data)   # N-vector
    W_out_sanger = real.(W_resp[1,:]); W_out_sanger ./= norm(W_out_sanger)  # N-vector

    y_pred_theory = W_out_theory'*X_test_theory
    y_pred_data   = W_out_data'*X_test_c
    y_pred_sanger = W_out_sanger'*X_test_c

    return y_pred_theory, y_pred_data, y_pred_sanger, y_test
end


T_input, T_relax = 100, 20
y_theory, y_data, y_sanger, y_test =
    run_fixate_response_readouts(N, 0.5, 0.9, T_input, T_relax)

# orient each readout so response 1 deflects upward (sign of a readout is arbitrary)
resp1 = (T_input+1):(T_input+T_relax)
fixall = vcat(1:T_input, (T_input+T_relax+1):(2T_input+T_relax))
function orient(y)
    y = vec(y)
    y .* sign(mean(y[resp1]) - mean(y[fixall]))
end
y_theory = orient(y_theory)
y_data   = orient(y_data)
y_sanger = orient(y_sanger)

# response windows for baseline (exclude the tail, as the fit does)
T_fix = vcat(1:T_input, (T_input+T_relax+1):(2T_input+T_relax))

# remove the first section of each fixation period, to avoid warm-up effects
remove_frac = 0.4
n_remove = round(Int, remove_frac * T_input)

function panel(y, ttl, c)
    y = vec(y)

    # Remove the warm-up portion of each fixation period
    keep = trues(length(y))
    keep[1:n_remove] .= false
    keep[(T_input + T_relax + 1):(T_input + T_relax + n_remove)] .= false

    y_plot = y[keep]
    x_plot = 1:length(y_plot)

    # Response windows after compression
    resp1_start = T_input - n_remove + 1
    resp1_end   = resp1_start + T_relax - 1

    resp2_start = resp1_end + (T_input - n_remove) + 1
    resp2_end   = resp2_start + T_relax - 1

    # Remaining fixation points
    fix1 = 1:(T_input - n_remove)
    fix2 = (resp1_end + 1):(resp2_start - 1)
    T_fix_plot = vcat(fix1, fix2)

    # Fixation mean
    fix_mean = mean(y_plot[T_fix_plot])

    p = plot(x_plot, y_plot,
             c=c, lw=2, label=false, title=ttl,
             grid=false, xlabel="Time", ylabel="Output",
             legend=false)

    # Response-period shading
    vspan!(p, [resp1_start, resp1_end],
           c=:grey, alpha=0.12, label=false)

    vspan!(p, [resp2_start, resp2_end],
           c=:grey, alpha=0.12, label=false)

    # Fixation mean
    hline!(p, [fix_mean],
           c=:black, ls=:dash, lw=1.2, label=false)

    # Annotations
    y_top = maximum(y_plot)
    y_bottom = minimum(y_plot)
    y_range = y_top - y_bottom

    annotate!(p,
        (mean(fix1), y_bottom + 0.08*y_range,
         text("Fixation", 9, :red)))

    annotate!(p,
        (mean(resp1_start:resp1_end), y_bottom+ 0.08*y_range,
         text("Response", 9, :red)))

         annotate!(p,
        (mean(fix2), y_top - 0.08*y_range,
         text("Fixation", 9, :red)))

    annotate!(p,
        (mean(resp2_start:resp2_end), y_top - 0.08*y_range,
         text("Response", 9, :red)))

    return p
end

plot(panel(y_theory, "Theory I",  :magenta3),
     panel(y_data,   "Theory II", :purple),
     panel(y_sanger, "Hebbian",   :teal),
     layout=(3, 1), size=(800, 600),
     yticks=false, leftmargin=10mm)

