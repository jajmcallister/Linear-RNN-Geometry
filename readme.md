# Linear-RNN-Geometry

Code for the preprint:

> **Geometry of linear recurrent neural network dynamics for sequence working memory, signal discrimination, and response suppression**
> James McAllister and Cian O'Donnell

We study linear RNNs with fixed, untrained recurrent weights. Their fixed points and stationary covariance have closed-form solutions, which we use to understand how three cognitive tasks can be solved from the geometry of network activity. Readouts are either built analytically or learned with a local Hebbian rule (Sanger's rule).

## Files

| File | Contents | Paper |
|---|---|---|
| `1.functions.jl` | Shared functions: random reservoir, linear RNN simulation, fixed points, stationary covariance, Sanger's rule | Sec. II |
| `2.sequence_working_memory.jl` | Lag Gram matrix, lag–PC correlations, theory and Hebbian readouts, input alignment | Sec. III, Fig. 2 |
| `3.signal_separation.jl` | Rank-1 and full-spectrum gain, empirical d′² vs theory, Fisher vs Hebbian decision readouts | Sec. IV, Figs. 4–5, App. C |
| `4.fixation_response.jl` | Fixation-response task with Theory I, Theory II and Hebbian readouts | Sec. V, Fig. 6 |

Scripts 2–4 load `1.functions.jl`, so keep all files in the same folder.

## Requirements

Julia v1.12.5 with these packages:

```julia
using Pkg
Pkg.add(["MultivariateStats", "Distributions", "Plots", "MatrixEquations"])
```

`LinearAlgebra`, `Random` and `Statistics` come with Julia.

## Usage

Run a script from its folder:

```julia
include("2.sequence_working_memory.jl")
```

or step through it in an editor such as VS Code. Each script makes the plots for its section of the paper.

## Notes

- Some scripts use fewer trials than the paper so they run quickly. Increase the trial counts to reproduce the averaged figures.
- The Hebbian readouts depend on the learning rate `η` and `epochs`.
