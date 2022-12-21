################################################################################
import Pkg
cd(@__DIR__)
Pkg.activate(".")
Pkg.status()
Pkg.instantiate()

using Plots, Random, Distributions, BenchmarkTools, UnPack, ArgCheck
using ModelWrappers, BaytesFilters, BaytesSMC, Baytes, BaytesInference

include("src/models/core/core.jl")

################################################################################
#constants
import BaytesCore: ByRows, ByCols
n = 1000
latent_type = Int32
cvg = 1.0 #pf coverage
_rng = Random.Xoshiro(1)

_N1 = 500
_Npart = 100
_TuningIter = 400

plot_default_color = :rainbow_bgyrm_35_85_c71_n256
plot_default_size = (1000,1000)
_fontsize = 16
_axissize = 16

################################################################################
################################################################################
################################################################################
# Chapter 3

################################################################################
# VIX data for Application
using JLD2
pwd()
dat_fin = jldopen( string(pwd(), "/src/data/vix_data.jld2" ) )
data_fin = read(dat_fin, "log_vix")
data_fin = data_fin[(end-n+1):end]
dates_fin = read(dat_fin, "dates")[(end-n+1):end]

plot_data = plot(
    size = plot_default_size, foreground_color_legend = nothing, legend=false,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize, margin=10Plots.mm
)
plot!(dates_fin, data_fin, xlabel = "Time", ylabel = "VIX Index Log Scale", legend=false)
#Plots.savefig("Chp6_realdata.png")

################################################################################
# Particle Filter - state trajectory
include("src/models/hsmm.jl")

data_og = deepcopy(data_HSMM_UV)
latent_og = deepcopy(latent_HSMM_UV)
model_og = deepcopy(hsmm_UV)
fieldnames( typeof(model_og.val))
taggedᵐᶜᵐᶜ = Tagged(model_og, (:μ, :σ, :r, :ϕ))
taggedᵖᶠ   = Tagged(model_og, :latent )

plot_hsmm = plot(layout=(3,1),
    size = plot_default_size,
    foreground_color_legend = nothing,
    #legend=false,
    background_color_legend = nothing,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize
)

plot!(data_og, ylabel="Generated data", label=false, subplot=1)
plot!(getfield.(latent_og, 1), ylabel="Latent state", label=false, subplot=2)
plot!(getfield.(latent_og, 2), ylabel="Duration in current state", label=false, subplot=3)
xaxis!("Time", subplot=3)
#Plots.savefig("Chp5_HSMM.png")

#Add PF estimate
objectiveᵖᶠ = Objective(deepcopy(model_og), data_og, taggedᵖᶠ)

pf = ParticleFilter(_rng, deepcopy(objectiveᵖᶠ),
    ParticleFilterDefault(; referencing=Marginal(), coverage = 1.0, threshold = 1.0)
)
val, diag = propose(_rng, pf, objectiveᵖᶠ)
plot!(getfield.(val.latent, 1), label="PF Latent state", subplot=2)
plot!(getfield.(val.latent, 2), label="PF Duration in current state", subplot=3)
Plots.savefig("Chp5_HSMM_PF.png")

################################################################################
# Particle Filter - likelihood estimation
include("src/models/hsmm.jl")

model_og
θ = (;
    μ = -5.:.05:1.,
    σ = 1:.05:7.,
    r = 1:.5:25,
    ϕ =.1:.005:.5,
)

# LL with few particles
objectiveᵖᶠ.model
pf = ParticleFilter(_rng, objectiveᵖᶠ, ParticleFilterDefault(;coverage = .10))
_plt1, loglik_pf1, loglik_exact1 = check_pf(_rng, pf, objectiveᵖᶠ, θ; dmax = 500, exact=true)
_plt1

# LL with many particles
pf = ParticleFilter(_rng, objectiveᵖᶠ, ParticleFilterDefault(;coverage = .50))
_plt2, loglik_pf2, loglik_exact2 = check_pf(_rng, pf, objectiveᵖᶠ, θ; exact=false, dmax = 500)
_plt2
# LL with many particles
pf = ParticleFilter(_rng, objectiveᵖᶠ, ParticleFilterDefault(;coverage = 1.00))
_plt22, loglik_pf22, loglik_exact22 = check_pf(_rng, pf, objectiveᵖᶠ, θ; exact=false, dmax = 500)
_plt22
# LL with many particles
pf = ParticleFilter(_rng, objectiveᵖᶠ, ParticleFilterDefault(;coverage = 2.0))
_plt3, loglik_pf3, loglik_exact3 = check_pf(_rng, pf, objectiveᵖᶠ, θ; exact=false, dmax = 500)
_plt3

#Now plot variance of pf for various number of particles:
loglik_all_pf = [loglik_pf1, loglik_pf2, loglik_pf22, loglik_pf3]
loglik_exact = loglik_exact1
Nparticles = Int64.([n*.1, n*.5, n*1., n*2.])
param_true = [-2., 4., 10., .3]

plot_ll = plot(
    layout=( size(loglik_all_pf[1], 1), 1 ),
    foreground_color_legend = nothing, legend=:topleft,
    #    ylabel="Log likelihood",
        size = plot_default_size,
        titlefontsize = 26,
        xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
        xtickfontsize=_axissize, ytickfontsize=_axissize
)
Plots.plot!(title=
    string(
    "PF log-likelihood estimates for 1000 data points, \n ", Nparticles[1],
    " (blue), ", Nparticles[2],
    " (green), ", Nparticles[3],
    " (yellow), ", Nparticles[4], "(orange) particles.         "), subplot=1,
    titleloc = :bottom, titlefont = font(11)
)

plot_ll
for (idx, sym) in enumerate(keys(θ))
    for iter in eachindex(loglik_all_pf)
        Plots.scatter!(θ[idx], loglik_all_pf[iter][idx]',
            label = "", #string(Nparticles[iter], " particles"),
            titlefontsize = 20,
            ylabel="Log likelihood",
            markerstrokewidth=0.5, alpha = 1.0, shape=:o,
            markerstrokecolor=Plots.palette(:rainbow_bgyrm_35_85_c71_n256, (length( keys(θ) )+1) )[iter],#"grey",
            markersize = 3,
            color = Plots.palette(:rainbow_bgyrm_35_85_c71_n256, (length( keys(θ) )+1) )[iter],
            subplot = idx
        )
    end
    plot!([param_true[idx]], seriestype = :vline, subplot = idx, label=false, color="grey")
    xaxis!(string("Range for ", sym), subplot=idx)
end
plot_ll
#change based on credible intervals
plot!(ylim=(-2600, -2450), xlim=(-3.0, -1.1), subplot=1)
plot!(ylim=(-2600, -2450), xlim=(3.0, 5.1), subplot=2)
plot!(ylim=(-2600, -2450), xlim=(5.0, 15.1), subplot=3)
plot!(ylim=(-2600, -2450), xlim=(0.2, 0.410), subplot=4)
plot_ll
Plots.savefig("Chp3_HSMM_ParticleFilterEstimate.pdf")

using JLD2
@save "pf_estimates.jld2" loglik_all_pf loglik_exact Nparticles param_true θ

#=
pf_estimates = jldopen( string(pwd(), "/src/data/pf_estimates.jld2" ) )

loglik_all_pf =  read(pf_estimates, "loglik_all_pf")
loglik_exact =  read(pf_estimates, "loglik_exact")
Nparticles  =  read(pf_estimates, "Nparticles")
param_true  =  read(pf_estimates, "param_true")
θ =  read(pf_estimates, "θ")
=#

################################################################################
################################################################################
################################################################################
# Chapter 4

################################################################################
# SECTION Simulation - Basic estimation
include("src/models/hsmm.jl")

modelᵗᵉᵐᵖ = deepcopy(hsmm_UV)
dataᵗᵉᵐᵖ = data_HSMM_UV
_pmcmc2 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage=1.0),
    NUTS( (:μ, :σ, :r, :ϕ); init=PriorInitialization())
)
trace_pmcmc2, algorithm_pmcmc2 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc2;
    default = SampleDefault(;
    printoutput = true, safeoutput = false,
    iterations = 2000, chains = 4, burnin = 1000,
    report = ProgressReport(; bar = true, log = SilentLog()))
)
#PLOT:

tar = Tagged(modelᵗᵉᵐᵖ, _pmcmc2.mcmc.sym)
tar2 = Tagged(modelᵗᵉᵐᵖ, :latent)
BaytesInference.plotChain(trace_pmcmc2, tar; model = hsmm_UV, burnin=000)
plotLatent(trace_pmcmc2, tar2; data = data_HSMM_UV, latent = latent_HSMM_UV)

using JLD2
@save "pmcmc_generated.jld2" trace_pmcmc2 algorithm_pmcmc2 modelᵗᵉᵐᵖ data_HSMM_UV  latent_HSMM_UV

BaytesInference.plotChain(trace_pmcmc2, tar;
    burnin = 000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp5_HSMM_PMCMC_CHAIN.png")

BaytesInference.plotLatent(trace_pmcmc2, tar2;
    data = data_HSMM_UV,
    burnin = 000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp5_HSMM_PMCMC_LATENT.png")

chainsummary(trace_pmcmc2, modelᵗᵉᵐᵖ, keys(tar.parameter), Val(:latex), 1000, 1, PrintDefault())

N1 = 250
_smc2 = SMC2(ParticleFilter(:latent; coverage = 1.0),
    ParticleGibbs(ParticleFilter(:latent; referencing = Ancestral(), coverage = cvg),
                  MCMC(NUTS, (:μ, :σ, :r, :ϕ); init=PriorInitialization())
                  ); resamplingthreshold = 0.75
)

Nchains = 100
trace_smc1, algorithm_smc1 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _smc2;
    default = SampleDefault(;
    printoutput = true, dataformat = Expanding(N1), iterations = 750, chains = Nchains, burnin = 200,
    report = ProgressReport(; bar = true, log = SilentLog()))
)

using JLD2
@save "smc_generated.jld2" trace_smc1 algorithm_smc1 modelᵗᵉᵐᵖ data_HSMM_UV  latent_HSMM_UV

_burnin = 400
transform_smc_gen = Baytes.TraceTransform(trace_smc1, modelᵗᵉᵐᵖ, algorithm_smc1.particles.kernel[1].pmcmc.kernel.mcmc.tune.tagged,
    TransformInfo(collect(1:trace_smc1.info.sampling.Nchains), [1], _burnin:1:trace_smc1.info.sampling.iterations)
)
summary(trace_smc1, algorithm_smc1, transform_smc_gen)

BaytesInference.plotChain(trace_smc1, tar;
    burnin = 000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp5_HSMM_SMC_CHAIN.png")

BaytesInference.plotLatent(trace_smc1, tar2;
    data = data_HSMM_UV,
    burnin = 250,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp5_HSMM_SMC_LATENT.png")

BaytesInference.plotDiagnostics(trace_smc1.diagnostics, algorithm_smc1;
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp5_HSMM_SMC_DIAGNOSTICS.png")

BaytesInference.plotPosteriorPrediction(trace_smc1.diagnostics, BaytesInference._SMC2();
    data = data_HSMM_UV,
    burnin=000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    CIRegion = [.025, .975]
)
Plots.savefig("Chp5_HSMM_SMC_PREDICT.png")



################################################################################
# SECTION Simulation - Check ACF function for various number of particles
include("src/models/hsmm.jl")

modelᵗᵉᵐᵖ = deepcopy(hsmm_UV)
dataᵗᵉᵐᵖ = data_HSMM_UV
# 100 particles
_pmcmc2_100 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage=0.1),
    NUTS( (:μ, :σ, :r, :ϕ); init=NoInitialization())
)
trace_pmcmc2_100, algorithm_pmcmc2_100 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc2_100;
    default = SampleDefault(;
    printoutput = true, safeoutput = false,
    iterations = 2000, chains = 4, burnin = 1000,
    report = ProgressReport(; bar = true, log = SilentLog()))
)
tar = Tagged(modelᵗᵉᵐᵖ, _pmcmc2_100.mcmc.sym)
tar2 = Tagged(modelᵗᵉᵐᵖ, :latent)
BaytesInference.plotChain(trace_pmcmc2_100, tar; model = hsmm_UV, burnin=000)
plotLatent(trace_pmcmc2_100, tar2; data = data_HSMM_UV, latent = latent_HSMM_UV)

# 500 particles
_pmcmc2_500 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage=0.5),
    NUTS( (:μ, :σ, :r, :ϕ); init=NoInitialization())
)
trace_pmcmc2_500, algorithm_pmcmc2_500 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc2_500;
    default = SampleDefault(;
    printoutput = true, safeoutput = false,
    iterations = 2000, chains = 4, burnin = 1000,
    report = ProgressReport(; bar = true, log = SilentLog()))
)
tar = Tagged(modelᵗᵉᵐᵖ, _pmcmc2_500.mcmc.sym)
tar2 = Tagged(modelᵗᵉᵐᵖ, :latent)
BaytesInference.plotChain(trace_pmcmc2_500, tar; model = hsmm_UV, burnin=000)
plotLatent(trace_pmcmc2_500, tar2; data = data_HSMM_UV, latent = latent_HSMM_UV)

# 1000 particles
_pmcmc2_1000 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage=1.0),
    NUTS( (:μ, :σ, :r, :ϕ); init=NoInitialization())
)
trace_pmcmc2_1000, algorithm_pmcmc2_1000 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc2_1000;
    default = SampleDefault(;
    printoutput = true, safeoutput = false,
    iterations = 2000, chains = 4, burnin = 1000,
    report = ProgressReport(; bar = true, log = SilentLog()))
)
tar = Tagged(modelᵗᵉᵐᵖ, _pmcmc2_1000.mcmc.sym)
tar2 = Tagged(modelᵗᵉᵐᵖ, :latent)
BaytesInference.plotChain(trace_pmcmc2_1000, tar; model = hsmm_UV, burnin=000)
plotLatent(trace_pmcmc2_1000, tar2; data = data_HSMM_UV, latent = latent_HSMM_UV)

# 2000 particles
_pmcmc2_2000 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage=2.0),
    NUTS( (:μ, :σ, :r, :ϕ); init=NoInitialization())
)
trace_pmcmc2_2000, algorithm_pmcmc2_2000 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc2_2000;
    default = SampleDefault(;
    printoutput = true, safeoutput = false,
    iterations = 2000, chains = 4, burnin = 1000,
    report = ProgressReport(; bar = true, log = SilentLog()))
)
tar = Tagged(modelᵗᵉᵐᵖ, _pmcmc2_2000.mcmc.sym)
tar2 = Tagged(modelᵗᵉᵐᵖ, :latent)
BaytesInference.plotChain(trace_pmcmc2_2000, tar; model = hsmm_UV, burnin=000)
plotLatent(trace_pmcmc2_2000, tar2; data = data_HSMM_UV, latent = latent_HSMM_UV)

# Grab likelihood estimates
using StatsBase
_effective_iter = 1000:2000
ll_100 = [ [trace_pmcmc2_100.diagnostics[Nchain][1][_effective_iter][iter].base.ℓobjective for iter in eachindex(trace_pmcmc2_100.diagnostics[Nchain][1][_effective_iter]) ] for Nchain in eachindex(trace_pmcmc2_100.diagnostics) ]
ll_500 = [ [trace_pmcmc2_500.diagnostics[Nchain][1][_effective_iter][iter].base.ℓobjective for iter in eachindex(trace_pmcmc2_500.diagnostics[Nchain][1][_effective_iter]) ] for Nchain in eachindex(trace_pmcmc2_500.diagnostics) ]
ll_1000 = [ [trace_pmcmc2_1000.diagnostics[Nchain][1][_effective_iter][iter].base.ℓobjective for iter in eachindex(trace_pmcmc2_1000.diagnostics[Nchain][1][_effective_iter]) ] for Nchain in eachindex(trace_pmcmc2_1000.diagnostics) ]
ll_2000 = [ [trace_pmcmc2_2000.diagnostics[Nchain][1][_effective_iter][iter].base.ℓobjective for iter in eachindex(trace_pmcmc2_2000.diagnostics[Nchain][1][_effective_iter]) ] for Nchain in eachindex(trace_pmcmc2_2000.diagnostics) ]

pacf_100 = StatsBase.pacf(reduce(vcat, ll_100), collect(1:10))
pacf_500 = StatsBase.pacf(reduce(vcat, ll_500), collect(1:10))
pacf_1000 = StatsBase.pacf(reduce(vcat, ll_1000), collect(1:10))
pacf_2000 = StatsBase.pacf(reduce(vcat, ll_2000), collect(1:10))

#Make summary plot
plot_ll = plot(
    layout=( 4, 1 ),
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    legend=:topright,
    #    ylabel="Log likelihood",
        size = plot_default_size,
        titlefontsize = 26,
        xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
        xtickfontsize=_axissize, ytickfontsize=_axissize
)
plot!(pacf_100, label="100 Particles", ylabel = "PACF", subplot=1)
plot!(pacf_500, label="500 Particles", ylabel = "PACF", subplot=2)
plot!(pacf_1000, label="1000 Particles", ylabel = "PACF", subplot=3)
plot!(pacf_2000, label="2000 Particles", ylabel = "PACF", xlabel="Lag", subplot=4)

plot_ll = plot(
    layout=( 1, 1 ),
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    legend=:topright,
    #    ylabel="Log likelihood",
        size = plot_default_size,
        titlefontsize = 26,
        xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
        xtickfontsize=_axissize, ytickfontsize=_axissize
)
_lw=2
plot!(pacf_100, label="100 Particles", linewidth=_lw, ylabel = "PACF", subplot=1)
plot!(pacf_500, label="500 Particles", linewidth=_lw, ylabel = "PACF", subplot=1)
plot!(pacf_1000, label="1000 Particles", linewidth=_lw, ylabel = "PACF", subplot=1)
plot!(pacf_2000, label="2000 Particles", linewidth=_lw, ylabel = "PACF", xlabel="Lag", subplot=1)
Plots.savefig("Chp3_PF_PACF.pdf")

#save
using JLD2
@save "pmcmc_pacf.jld2" trace_pmcmc2_100 trace_pmcmc2_500 trace_pmcmc2_1000 trace_pmcmc2_2000  modelᵗᵉᵐᵖ data_HSMM_UV  latent_HSMM_UV

################################################################################
################################################################################
################################################################################
# Chapter 5

################################################################################
# SECTION Applications - PART 1 - Determining the number of hiddens states in HSMM

#Experiment - estimate 3 state HSMM with 2 and 5 state HSMM -> check performance
include("src/models/hsmm.jl")
#_rng = Xoshiro(1)
ntrials = 1000
latent_type = Int32
_Ninitial = 500
_Nchains = 100
_Burnin = 100
################First, create HSMMs

##### 2 state
latent_init2 = [( convert(latent_type, rand(Categorical(2)) ), convert(latent_type, rand(Poisson(10) ) ) ) for iter in 1:ntrials]
param2 = (;
    μ = Param([-2., 2.], [truncated(Normal(-0.1, 10^5), -10., 0.0), truncated(Normal(0.1, 10^5), 0.0, 10.0) ]),
    σ = Param([2.5, 0.5], [truncated(Normal(2., 10^5), 0.0, 10.0), truncated(Normal(1., 10^5), 0.0, 10.0)]), #[Gamma(1.,1.) for i in 1:3]),
    λ = Param([10., 50.], [truncated(Normal(10.0, 10^5), 0.0, 11.0), truncated(Normal(50.0, 10^5), 0.0, 100.0)]),
    p = Param([[1], [1]], Fixed()),
    latent = Param(latent_init2, Fixed()),
)
model2 = ModelWrapper(HSMM_UV_P(), param2)
data2, latent2 = simulate(_rng, model2; Nsamples = ntrials)
_tagged2 = Tagged(model2, :latent)
fill!(model2, _tagged2, (; latent = latent2))

##### 3 state
latent_init3 = [( convert(latent_type, rand(Categorical(3)) ), convert(latent_type, rand(Poisson(10) ) ) ) for iter in 1:ntrials]
param3 = (;
    μ = Param([-5., 0.0, 5.],
        [truncated(Normal(-1., 10^5), -10., -0.01), truncated(Normal(0., 10^5), -10., 10.), truncated(Normal(1., 10^5), 00.1, 10.0) ]),
    σ = Param([2.5, 1.5, 0.5],
        [truncated(Normal(2.5, 10^5), 0.0, 10.0),  truncated(Normal(1.5, 10^5), 0.0, 10.0), truncated(Normal(0.5, 10^5), 0.0, 10.0)]),
    λ = Param([5., 10., 30.],
    [truncated(Normal(5.0, 10^5), 0.0, 10.0),truncated(Normal(10.0, 10^5), 0.0, 30.0), truncated(Normal(30.0, 10^5), 0.0, 100.0)]),
    p = Param([[.2, .8], [.2, .8], [.2, .8]],
        [Dirichlet(2,2), Dirichlet(2,2), Dirichlet(2,2)]),
    latent = Param(latent_init3, Fixed()),
)
model3 = ModelWrapper(HSMM_UV_P5(), param3)
data3, latent3 = simulate(_rng, model3; Nsamples = ntrials)
_tagged3 = Tagged(model3, :latent)
fill!(model3, _tagged3, (; latent = latent3))

using JLD2
@save "nstates_modeldata.jld2" model3 data3 latent3

##### 5 state
latent_init5 = [( convert(latent_type, rand(Categorical(5)) ), convert(latent_type, rand(Poisson(10) ) ) ) for iter in 1:ntrials]
param5 = (;
    μ = Param([-5., -2.5, 0.0, 2.5,  5.],
        [truncated(Normal(-5., 10^5), -10., -0.01), truncated(Normal(0., 10^5), -10., 10.), truncated(Normal(0., 10^5), -10., 10.), truncated(Normal(0., 10^5), -10., 10.), truncated(Normal(5., 10^5), 00.1, 10.0) ]),
    σ = Param([2.5, 1.5, 0.5, 0.5, 0.5],
        [truncated(Normal(2.5, 10^5), 0.1, 10.0),  truncated(Normal(1.5, 10^5), 0.1, 10.0), truncated(Normal(0.5, 10^5), 0.1, 10.0), truncated(Normal(0.5, 10^5), 0.1, 10.0), truncated(Normal(0.5, 10^5), 0.1, 10.0)]),
    λ = Param([5., 10., 10., 10., 30.],
    [truncated(Normal(5.0, 10^5), 1.0, 10.0), truncated(Normal(10.0, 10^5), 1.0, 30.0), truncated(Normal(10.0, 10^5), 1.0, 30.0), truncated(Normal(10.0, 10^5), 1.0, 30.0), truncated(Normal(30.0, 10^5), 1.0, 100.0)]),
    p = Param([[.2, .2, .2, .4], [.2, .2, .2, .4], [.2, .2, .2, .4], [.2, .2, .2, .4], [.2, .2, .2, .4]],
        [Dirichlet(4,1/4), Dirichlet(4,1/4), Dirichlet(4,1/4), Dirichlet(4,1/4), Dirichlet(4,1/4)]),
    latent = Param(latent_init5, Fixed()),
)
model5 = ModelWrapper(HSMM_UV_P5(), param5)
data5, latent5 = simulate(_rng, model5; Nsamples = ntrials)
_tagged5 = Tagged(model5, :latent)
fill!(model5, _tagged5, (; latent = latent5))
model5.val.latent
_rng

######################################
# Estimate Models
#1 recover model parameter with model3
_pmcmc3 = ParticleGibbs(ParticleFilter(:latent; referencing=Ancestral(), coverage = 1.5), NUTS( (:μ, :σ, :λ, :p)))
_tagged33 = Tagged(model3, (:μ, :σ, :λ, :p))
_smc3 = SMC2(ParticleFilter(:latent, coverage = 1.0), _pmcmc3)
# Single Chain
trace_smc3, algorithm_smc3 = sample(_rng, deepcopy(model3), data3, _smc3;
    default = SampleDefault(;
    dataformat = Expanding(_Ninitial),
    chains = _Nchains, burnin = _Burnin,
    printoutput = true,
    safeoutput  = true
    )
)
plotChain(trace_smc3, _tagged33; model = model3)
plotDiagnostics(trace_smc3.diagnostics, algorithm_smc3)
plotLatent(trace_smc3, _tagged3; data = data3, latent = latent3)
plotPosteriorPrediction(trace_smc3.diagnostics, _SMC2(); data = data3, latent = latent3, burnin=00)

#Check how Model 2 does
_pmcmc2 = ParticleGibbs(ParticleFilter(:latent; referencing=Ancestral(), coverage = 1.0), NUTS( (:μ, :σ, :λ)))
_tagged22 = Tagged(model2, (:μ, :σ, :λ))
_smc2 = SMC2(ParticleFilter(:latent, coverage = 1.0), _pmcmc2)
# Single Chain
trace_smc2, algorithm_smc2 = sample(_rng, deepcopy(model2), data3, _smc2;
    default = SampleDefault(;
    dataformat = Expanding(_Ninitial),
    chains = _Nchains, burnin = _Burnin,
    printoutput = true,
    safeoutput  = true
    )
)
plotChain(trace_smc2, _tagged22)
plotDiagnostics(trace_smc2.diagnostics, algorithm_smc2)
plotLatent(trace_smc2, _tagged2; data = data3, latent = latent3)
plotPosteriorPrediction(trace_smc2.diagnostics, _SMC2(); data = data3, latent = latent3, burnin=00)

#Check how Model 5 does
_pmcmc5 = ParticleGibbs(ParticleFilter(:latent; referencing=Ancestral(), coverage = 2.0), NUTS( (:μ, :σ, :λ, :p)))
_tagged55 = Tagged(model5, (:μ, :σ, :λ, :p))
_smc5 = SMC2(ParticleFilter(:latent, coverage = 2.0), _pmcmc5)
# Single Chain
trace_smc5, algorithm_smc5 = sample(_rng, deepcopy(model5), data3, _smc5;
    default = SampleDefault(;
    dataformat = Expanding(_Ninitial),
    chains = _Nchains, burnin = _Burnin,
    printoutput = true,
    safeoutput  = true
    )
)
transform_real = Baytes.TraceTransform(trace_smc5, model5, _tagged55,
    TransformInfo(collect(1:trace_smc5.info.sampling.Nchains), [1], 1:1:trace_smc5.info.sampling.iterations)
)
transform_real
chainsummary(trace_smc5, transform_real, Val(:text), PrintDefault(; Ndigits=2, quantiles=[0.025, 0.25, 0.50, 0.75, 0.975]) )
#Baytes.savetrace(trace_smc5, model5, algorithm_smc5)

plotChain(trace_smc5, _tagged55)
plotDiagnostics(trace_smc5.diagnostics, algorithm_smc5)
plotLatent(trace_smc5, _tagged5; data = data3, latent = latent3)
plotPosteriorPrediction(trace_smc5.diagnostics, _SMC2(); data = data3, latent = latent3, burnin=00)
Plots.savefig("HSMM 5 state detection - prediction.png")

################################################################################
################################################################################
################################################################################
# LOG VIX Application
_N1 = 500
_Npart = 100
_TuningIter = 50
_burnin = 0

using JLD2
dat_fin = jldopen(string(pwd(), "/src/data/vix_data.jld2") )
data_fin = read(dat_fin, "log_vix")
data_fin = data_fin[(end-n+1):end]
dates_fin = read(dat_fin, "dates")[(end-n+1):end]

plot_data = plot(
    size = plot_default_size, foreground_color_legend = nothing, legend=false,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize, margin=10Plots.mm
)
plot!(dates_fin, data_fin, xlabel = "Time", ylabel = "VIX Index Log Scale", legend=false)


################################################################################
###############################################################################
## AR HSMM with Neg Binomial duration
include("src/models/arhsmm_2state.jl")

param = (   μ = Param([2.0, 0.5],
                    [truncated(Normal(2.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)]),
                    σ = Param([.2, .1],
                    [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)]),
                    w = Param([.2, .8],
                    [truncated(Normal(0.0,10^5.), 0.0, .7), truncated(Normal(0.8,10^5.), 0.0, 1.)]),
                    r = Param([5., 2.5],
                        [truncated(Normal(10.0, 10^5), 0.0, 50.0), truncated(Normal(5.0, 10^5), 0.0, 20.0)]),
                    ϕ = Param([.1, .1],
                        [Beta(1., 1.), Beta(1., 1.)]),
                   latent = Param(latent_init, Fixed())
                )

model = ModelWrapper(ARHSMM2(), param)
data, latent = simulate(_rng, model;Nsamples = n)
_tagged = Tagged(model, :latent)
fill!(model, _tagged, (; latent = latent))
model.val.latent

################################################################################
# Testing
data_og    = deepcopy( data_fin )
latent_og  = deepcopy( latent)
model_og   = deepcopy( model )

fieldnames( typeof(model_og.val) )
taggedᵐᶜᵐᶜ = Tagged(model_og, (:μ, :σ, :w, :r, :ϕ))
taggedᵖᶠ   = Tagged(model_og, :latent )
length(taggedᵐᶜᵐᶜ)
################################################################################
## Make initial Model random
dataᵗᵉᵐᵖ    = deepcopy( data_og)
latentᵗᵉᵐᵖ  = deepcopy( latent_og)
modelᵗᵉᵐᵖ   = deepcopy( model_og)
objectiveᵐᶜᵐᶜ = Objective(deepcopy( model_og), dataᵗᵉᵐᵖ, taggedᵐᶜᵐᶜ)
objectiveᵖᶠ = Objective(deepcopy( model_og), dataᵗᵉᵐᵖ, taggedᵖᶠ)
objectiveᵐᶜᵐᶜ(model_og.val)
objectiveᵖᶠ(model_og.val)
plot(dataᵗᵉᵐᵖ)

_multi = 1
_pmcmc1 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage = cvg*_multi, init = OptimInitialization() ),
    NUTS((:μ, :σ, :w, :r, :ϕ); init=PriorInitialization(100) )
)
_smc = SMC2(ParticleFilter(:latent; coverage = cvg*_multi), _pmcmc1; Ntuning = _TuningIter)

#=
# Single Chain
trace_pmcmc_hsmm, algorithm_pmcmc_hsmm = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc1;
    default = SampleDefault(;chains = 4, iterations = 1000, burnin = 0, safeoutput  = false)
)
#PLOT:
plotChain(trace_pmcmc_hsmm, objectiveᵐᶜᵐᶜ.tagged; burnin = 0)
plotLatent(trace_pmcmc_hsmm, objectiveᵖᶠ.tagged; data = data_fin)
=#

## SMC
trace_smc_arhsmm2, algorithm_smc_arhsmm2 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _smc;
    default = SampleDefault(; dataformat = Expanding(_N1),
    chains = _Npart, burnin = 0, safeoutput  = true)
)

plotChain(trace_smc_arhsmm2, objectiveᵐᶜᵐᶜ.tagged;
    burnin=00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp6_arhsmm2_SMC_CHAIN.png")

plotLatent(trace_smc_arhsmm2,  objectiveᵖᶠ.tagged; data = data_og,
    burnin = 000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp6_arhsmm2_SMC_LATENT.png")

plotPosteriorPrediction(trace_smc_arhsmm2.diagnostics, _SMC2(); data = data_og,
    burnin=000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    CIRegion = [.1, .9]
)
Plots.savefig("Chp6_arhsmm2_SMC_PREDICT.png")
plotDiagnostics(trace_smc_arhsmm2.diagnostics, algorithm_smc_arhsmm2)
Plots.savefig("Chp6_arhsmm2_SMC_DIAGNOSTICS.png")
# Get back results and compute summary statistics and EScore

################################################################################
## AR HSMM with Poisson duration
include("models/arhsmm_2state.jl")

param = (   μ = Param([3.0, 1.0],
                    [truncated(Normal(3.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)]),
                    σ = Param([.2, .4],
                    [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)]),
                    w = Param([.2, .8],
                    [truncated(Normal(0.0,10^5.), -.5, .5), truncated(Normal(0.8,10^5.), 0.0, 1.)]),
                    λ = Param([30., 10.],
                        [truncated(Normal(30.0, 10^5), 0.0, 100.0), truncated(Normal(10.0, 10^5), 0.0, 30.0)]),
                   latent = Param(latent_init, Fixed())
                )
model = ModelWrapper(ARHSMMP2(), param)
data, latent = simulate(_rng, model; Nsamples = n)
_tagged = Tagged(model, :latent)
fill!(model, _tagged, (; latent = latent))
model.val.latent

################################################################################
# Testing
data_og    = deepcopy( data_fin )
latent_og  = deepcopy( latent)
model_og   = deepcopy( model )

fieldnames( typeof(model_og.val) )
taggedᵐᶜᵐᶜ = Tagged(model_og, (:μ, :σ, :w, :λ))
taggedᵖᶠ   = Tagged(model_og, :latent )
length(taggedᵐᶜᵐᶜ)
################################################################################
## Make initial Model random
dataᵗᵉᵐᵖ    = deepcopy( data_og)
latentᵗᵉᵐᵖ  = deepcopy( latent_og)
modelᵗᵉᵐᵖ   = deepcopy( model_og)
objectiveᵐᶜᵐᶜ = Objective(deepcopy( model_og), dataᵗᵉᵐᵖ, taggedᵐᶜᵐᶜ)
objectiveᵖᶠ = Objective(deepcopy( model_og), dataᵗᵉᵐᵖ, taggedᵖᶠ)
objectiveᵐᶜᵐᶜ(model_og.val)
objectiveᵖᶠ(model_og.val)

_multi = 1
_pmcmc1 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage = cvg*_multi, init = OptimInitialization()),
    NUTS((:μ, :σ, :w, :λ), init = PriorInitialization(1000))
)
_smc = SMC2(ParticleFilter(:latent; coverage = cvg*_multi), _pmcmc1; Ntuning = _TuningIter)

#=
# Single Chain
trace_pmcmc_hsmm, algorithm_pmcmc_hsmm = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc1;
    default = SampleDefault(;chains = 4, iterations = 500, burnin = 0, safeoutput  = false)
)
#PLOT:
plotChain(trace_pmcmc_hsmm, objectiveᵐᶜᵐᶜ.tagged; burnin = 0)
plotLatent(trace_pmcmc_hsmm, objectiveᵖᶠ.tagged; data = data_fin)
=#

## SMC
trace_smc_arhsmmP2, algorithm_smc_arhsmmP2 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _smc;
    default = SampleDefault(; dataformat = Expanding(_N1),
    chains = _Npart, burnin = 0, safeoutput  = true)
)
plotChain(trace_smc_arhsmmP2, objectiveᵐᶜᵐᶜ.tagged;
    burnin=00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp6_arhsmmP2_SMC_CHAIN.png")

plotLatent(trace_smc_arhsmmP2,  objectiveᵖᶠ.tagged; data = data_og,
    burnin = 00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp6_arhsmmP2_SMC_LATENT.png")

plotPosteriorPrediction(trace_smc_arhsmmP2.diagnostics, _SMC2(); data = data_og,
    burnin=000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    CIRegion = [.025, .975]
)
Plots.savefig("Chp6_arhsmmP2_SMC_PREDICT.png")
plotDiagnostics(trace_smc_arhsmmP2.diagnostics, algorithm_smc_arhsmmP2)
Plots.savefig("Chp6_arhsmmP2_SMC_DIAGNOSTICS.png")
# Get back results and compute summary statistics and EScore

################################################################################
################################################################################
# AR HMM

## AR HMM with
include("models/arhmm_2state.jl")

param = (   μ = Param([3.0, 1.0],
                    [truncated(Normal(3.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)]),
                    σ = Param([.2, .4],
                    [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)]),
                    w = Param([.2, .8],
                    [truncated(Normal(0.0,10^5.), -.5, .5), truncated(Normal(0.8,10^5.), 0.0, 1.)]),
                    p = Param([[.95, .05], [.7, .3]], [Dirichlet(2,2) for i in 1:2]),
                   latent = Param(latent_init, Fixed())
                )
model = ModelWrapper(ARHMM2(), param)
data, latent = simulate(_rng, model; Nsamples = n)
_tagged = Tagged(model, :latent)
fill!(model, _tagged, (; latent = latent))
model.val.latent

################################################################################
# Testing
data_og    = deepcopy( data_fin )
latent_og  = deepcopy( latent)
model_og   = deepcopy( model )

fieldnames( typeof(model_og.val) )
taggedᵐᶜᵐᶜ = Tagged(model_og, (:μ, :σ, :w, :p))
taggedᵖᶠ   = Tagged(model_og, :latent )
length(taggedᵐᶜᵐᶜ)
################################################################################
## Make initial Model random
dataᵗᵉᵐᵖ    = deepcopy( data_og)
latentᵗᵉᵐᵖ  = deepcopy( latent_og)
modelᵗᵉᵐᵖ   = deepcopy( model_og)
objectiveᵐᶜᵐᶜ = Objective(deepcopy( model_og), dataᵗᵉᵐᵖ, taggedᵐᶜᵐᶜ)
objectiveᵖᶠ = Objective(deepcopy( model_og), dataᵗᵉᵐᵖ, taggedᵖᶠ)
objectiveᵐᶜᵐᶜ(model_og.val)
objectiveᵖᶠ(model_og.val)

_multi = 2
_pmcmc1 = ParticleGibbs(
    ParticleFilter(:latent; referencing=Ancestral(), coverage = cvg*_multi, init = OptimInitialization()),
    NUTS((:μ, :σ, :w, :p), init = PriorInitialization(1000))
)
_smc = SMC2(ParticleFilter(:latent; coverage = cvg*_multi), _pmcmc1; Ntuning = _TuningIter)

#=
# Single Chain
trace_pmcmc_hsmm, algorithm_pmcmc_hsmm = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _pmcmc1;
    default = SampleDefault(;chains = 4, iterations = 500, burnin = _N1, safeoutput  = false)
)
#PLOT:
plotChain(trace_pmcmc_hsmm, objectiveᵐᶜᵐᶜ.tagged; burnin = 0)
plotLatent(trace_pmcmc_hsmm, objectiveᵖᶠ.tagged; data = data_fin)
=#

## SMC
trace_smc_arhmm2, algorithm_smc_arhmm2 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _smc;
    default = SampleDefault(; dataformat = Expanding(_N1),
    chains = _Npart, burnin = 0, safeoutput  = true)
)
plotChain(trace_smc_arhmm2, objectiveᵐᶜᵐᶜ.tagged;
    burnin=00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp6_arhmm2_SMC_CHAIN.png")

plotLatent(trace_smc_arhmm2,  objectiveᵖᶠ.tagged; data = data_og,
    burnin = 00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp6_arhmm2_SMC_LATENT.png")

plotPosteriorPrediction(trace_smc_arhmm2.diagnostics, _SMC2(); data = data_og,
    burnin=000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    CIRegion = [.025, .975]
)
Plots.savefig("Chp6_arhmm2_SMC_PREDICT.png")
plotDiagnostics(trace_smc_arhmm2.diagnostics, algorithm_smc_arhmm2)
Plots.savefig("Chp6_arhmm2_SMC_DIAGNOSTICS.png")
# Get back results and compute summary statistics and EScore

################################################################################
################################################################################
# AR(1) Model
include("models/AR1.jl")

param = (
    μ = Param(2.0, truncated(Normal(2.0,10^5.), 0., 10.)),
    σ = Param(.2, truncated(Normal(0.2,10^5.), 0.01, 5.)),
    w = Param(.8, truncated(Normal(0.0,10^5.), .0, 1.0)),
)
model = ModelWrapper(AR1(), param)
################################################################################
# Testing
data_og    = deepcopy( data_fin )
model_og   = deepcopy( model )

fieldnames( typeof(model_og.val) )
taggedᵐᶜᵐᶜ = Tagged(model_og, (:μ, :σ, :w))
length(taggedᵐᶜᵐᶜ)
################################################################################
## Make initial Model random
dataᵗᵉᵐᵖ    = deepcopy( data_og)
modelᵗᵉᵐᵖ   = deepcopy( model_og)
objectiveᵐᶜᵐᶜ = Objective(deepcopy( model_og), dataᵗᵉᵐᵖ, taggedᵐᶜᵐᶜ)
objectiveᵐᶜᵐᶜ(model_og.val)

_multi = 1
_mcmc1 = NUTS((:μ, :σ, :w), init = PriorInitialization(1000))
_smc = SMC(_mcmc1; Ntuning = _TuningIter)

#=
# Single Chain
trace_pmcmc_hsmm, algorithm_pmcmc_hsmm = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _mcmc1;
    default = SampleDefault(;chains = 4, iterations = 1000, burnin = _N1, safeoutput  = false)
)
#PLOT:
plotChain(trace_pmcmc_hsmm, objectiveᵐᶜᵐᶜ.tagged; burnin = 0)
=#

## SMC
trace_smc_ar1, algorithm_smc_ar1 = sample(_rng, modelᵗᵉᵐᵖ, dataᵗᵉᵐᵖ, _smc;
    default = SampleDefault(; dataformat = Expanding(_N1),
    chains = _Npart, burnin = 0, safeoutput  = true)
)

plotChain(trace_smc_ar1, objectiveᵐᶜᵐᶜ.tagged;
    burnin=00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp6_ar1_SMC_CHAIN.png")

plotPosteriorPrediction(trace_smc_ar1.diagnostics, _IBIS(); data = data_og,
    burnin=000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    CIRegion = [.025, .975]
)
Plots.savefig("Chp6_ar1_SMC_PREDICT.png")
plotDiagnostics(trace_smc_ar1.diagnostics, algorithm_smc_ar1)
Plots.savefig("Chp6_ar1_SMC_DIAGNOSTICS.png")
