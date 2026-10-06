using LinearAlgebra, Random, Statistics

# Random reservoir with spectral radius ρ. Entries iid N(0,1/N), then rescaled.
function random_reservoir(N, ρ)
    W = randn(N, N) / sqrt(N)
    W .*= ρ / maximum(abs.(eigvals(W)))
    return W
end

# Leaky linear RNN. input_seq is a vector (scalar input) or a T×d matrix.
function simulate_esn(W, w_in, input_seq, T, α)
    N = size(W, 1)
    X = zeros(N, T)
    x = zeros(N)
    for t in 1:T
        u = ndims(input_seq) == 1 ? input_seq[t] : vec(input_seq[t, :])
        x = (1 - α) * x + α * (W * x + w_in * u)
        X[:, t] = x
    end
    return X
end

function simulate_esn_ddm(W, w_in, input_seq, T, α, T_fixate, T_response)
    N = size(W, 1)
    X = zeros(N, T)
    x = zeros(N)
    for t in 1:T
        u = ndims(input_seq) == 1 ? input_seq[t] : vec(input_seq[t, :])
        x = (1 - α) * x + α * (W * x + w_in * u)
        if t % (T_fixate + T_response) == 0
            x .= 0.0  # reset state after each fixation-response cycle
        end
        X[:, t] = x
    end
    return X
end

W_eff(W, α) = (1 - α) * I + α * W

# Fixed point x* = (I - Weff)^{-1} α Win μ
fixed_point(W, w_in, μ, α) = (I - W_eff(W, α)) \ (α * w_in * μ)

# Stationary covariance from the discrete Lyapunov equation.
function state_covariance(W, w_in, Σin, α)
    Weff = W_eff(W, α)
    Q = α^2 * w_in * Σin * w_in'
    return lyapd(Weff, Matrix(Q)) # can use the discrete Lyapunov equation
end

# Sanger's rule / generalised Hebbian algorithm. Returns k×N readout,
# rows converging to the leading PCs of Cov[x].
function learn_pcs_sanger(X, k; η=1e-3, epochs=100)
    N, T = size(X)
    W = randn(k, N) ./ sqrt(N)
    for _ in 1:epochs
        for t in 1:T
            x = X[:, t]
            y = W * x
            for i in 1:k
                proj = zeros(N)
                for j in 1:i
                    proj .+= y[j] * W[j, :]
                end
                W[i, :] .+= η * y[i] * (x .- proj)
                W[i, :] ./= norm(W[i, :]) 
            end
        end
    end
    return W
end

# k lagged copies of a scalar input: row i is the input delayed by i steps.
function lagged_targets(u, k)
    T = length(u)
    targets = zeros(k, T)
    for i in 1:k
        targets[i, i+1:end] = u[1:T-i]
    end
    return targets
end
