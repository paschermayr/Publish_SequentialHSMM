################################################################################
import Pkg
cd(@__DIR__)
Pkg.activate(".")
Pkg.status()

#If environment activated for first time, uncomment next line to install all libraries used in project
#Pkg.instantiate()
include("preamble/_packages.jl");

################################################################################
# Set Preamble
include("preamble/_preamble.jl");

################################################################################
# Chapter 3

################################################################################
# VIX data for Application
_datayears = dates_fin[begin]:Dates.Day(30*6):dates_fin[end] #
_datayears = Date.(unique( Dates.format.(dates_fin, "yyyy") ))
_dataticks = Dates.format.(_datayears, "yyyy")#-mm

plot_data = plot(
    size = plot_default_size, foreground_color_legend = nothing, legend=false,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize, margin=10Plots.mm
)
plot!(dates_fin, data_fin, xlabel = "Time", ylabel = "VIX Index Log Scale", legend=false, xticks=(_datayears,_dataticks))
Plots.savefig("Chp6_realdata.png")

################################################################################
# Chapter 4 Simulated Data

#######################################
# PMCMC
f_model = jldopen(string(pwd(), "/output/Generated - PMCMC.jld2"));

trace_pmcmc_gen = read(f_model, "trace_pmcmc2");
algorithm_pmcmc_gen = read(f_model, "algorithm_pmcmc2");
model_gen = read(f_model, "modelᵗᵉᵐᵖ");
# saved in f_model
data_gen = read(f_model, "data_HSMM_UV");
latent_gen = read(f_model, "latent_HSMM_UV");

tar = algorithm_pmcmc_gen[1][1].tune.tagged
tar2 = Tagged(model_gen, :latent)
transform_pmcmc_gen = Baytes.TraceTransform(trace_pmcmc_gen, model_gen, tar, TransformInfo(collect(1:4), [1], 1:1:2000) )

BaytesInference.plotChain(trace_pmcmc_gen, tar;
    burnin = 000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
_fontsize2 = 16
plot!(xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize2
)
Plots.savefig("Chp5_HSMM_PMCMC_CHAIN.png")

BaytesInference.plotLatent(trace_pmcmc_gen, tar2;
    data = data_gen,
    latent = latent_gen,
    burnin = 000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp5_HSMM_PMCMC_LATENT.png")

summary(trace_pmcmc_gen, algorithm_pmcmc_gen, transform_pmcmc_gen)
chainsummary(trace_pmcmc_gen, transform_pmcmc_gen, Val(:text) )
chainsummary(trace_pmcmc_gen, transform_pmcmc_gen, Val(:latex) )

#######################################
# SMC
f_model = jldopen(string(pwd(), "/output/Generated - SMC.jld2"));

trace_smc_gen = read(f_model, "trace_smc1");
algorithm_smc_gen = read(f_model, "algorithm_smc1");
model_gen = read(f_model, "modelᵗᵉᵐᵖ");
data_gen = read(f_model, "data_HSMM_UV");
latent_gen = read(f_model, "latent_HSMM_UV");
trace_smc_gen.summary.info
tar = algorithm_smc_gen.particles.kernel[1].pmcmc.kernel.mcmc.tune.tagged
tar2 = Tagged(model_gen, :latent)
transform_smc_gen = Baytes.TraceTransform(trace_smc_gen, model_gen, tar,
    TransformInfo(collect(1:trace_smc_gen.summary.info.Nchains), [1], (1):1:trace_smc_gen.summary.info.iterations)
)
summary(trace_smc_gen, algorithm_smc_gen, transform_smc_gen)
printchainsummary(trace_smc_gen, transform_smc_gen, Val(:text) )
chainsummary(trace_smc_gen, transform_smc_gen)

BaytesInference.plotChain(trace_smc_gen, tar;
#    _xaxis = dates_fin,
    burnin = 000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
_fontsize2 = 14
plot!(xguidefontsize=_fontsize2, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize2
)
Plots.savefig("Chp5_HSMM_SMC_CHAIN.png")

BaytesInference.plotLatent(trace_smc_gen, tar2;
#    _xaxis = dates_fin,
    data = data_gen,
    latent = latent_gen,
    burnin = 250,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
Plots.savefig("Chp5_HSMM_SMC_LATENT.png")

BaytesInference.plotDiagnostics(trace_smc_gen.diagnostics, algorithm_smc_gen;
#    _xaxis = dates_fin,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
_fontsize2 = 12
plot!(xguidefontsize=_fontsize, yguidefontsize=_fontsize2, legendfontsize=_fontsize2,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize2
)
Plots.savefig("Chp5_HSMM_SMC_DIAGNOSTICS.png")

BaytesInference.plotPosteriorPrediction(trace_smc_gen.diagnostics, BaytesInference._SMC2();
#    _xaxis = dates_fin,
    data = data_gen,
#    latent = latent_gen,
    burnin=000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    CIRegion = [.025, .975]
)

_fontsize2 = 16
plot!(xguidefontsize=_fontsize2, yguidefontsize=_fontsize2, legendfontsize=_fontsize2,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize2
)
Plots.savefig("Chp5_HSMM_SMC_PREDICT.png")

################################################################################
# Particle Filter - state trajectory
data_og = deepcopy(data_gen)
latent_og = deepcopy(latent_gen)
model_og = deepcopy(model_gen)
fieldnames( typeof(model_og.val))
taggedᵖᶠ   = Tagged(model_og, :latent )

plot_hsmm = plot(layout=(3,1),
    size = plot_default_size,
    #legend=false,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize
)

plot!(data_og, ylabel="Generated data", label=false, subplot=1)
plot!(getfield.(latent_og, 1), ylabel="Latent state", label=false, subplot=2)
plot!(getfield.(latent_og, 2), ylabel="Duration in current state", label=false, subplot=3)
xaxis!("Time", subplot=3)

#Add PF estimate
objectiveᵖᶠ = Objective(deepcopy(model_og), data_og, taggedᵖᶠ)

pf = ParticleFilter(_rng, deepcopy(objectiveᵖᶠ),
    ParticleFilterDefault(; referencing=Marginal(), coverage = 1.0, threshold = 1.0)
)
val, diag = propose(_rng, dynamics(objectiveᵖᶠ), pf, objectiveᵖᶠ)
plot!(getfield.(val.latent, 1), label="PF Latent state", subplot=2)
plot!(getfield.(val.latent, 2), label="PF Duration in current state", subplot=3)
Plots.savefig("Chp5_HSMM_PF.png")

################################################################################
# Chapter 5 -

#######################################
# Determine number of states
f_stuff = jldopen(string(pwd(), "/output/Generated State Recovery - 3 state data.jld2"));

f_nstates_model = read(f_stuff, "model3");
f_nstates_data = read(f_stuff, "data3");
f_nstates_latent = read(f_stuff, "latent3");
f_nstates_model2 = jldopen(string(pwd(), "/output/Generated State Recovery - SMC 2 state Model - Trace.jld2"));
f_nstates_model3 = jldopen(string(pwd(), "/output/Generated State Recovery - SMC 3 state Model - Trace.jld2"));
f_nstates_model5 = jldopen(string(pwd(), "/output/Generated State Recovery - SMC 5 state Model - Trace.jld2"));

trace_nstates_saved2 = read(f_nstates_model2, "trace");
trace_nstates_saved3 = read(f_nstates_model3, "trace");
trace_nstates_saved5 = read(f_nstates_model5, "trace");

model_nstates_3 = read(f_nstates_model3, "model");
model_nstates_5 = read(f_nstates_model5, "model");

algorithm_nstates_3 = read(f_nstates_model3, "algorithm");
algorithm_nstates_5 = read(f_nstates_model5, "algorithm");

allnames = ["2 state HSMM", "3 state HSMM", "5 state HSMM"];
Nmodels = 5
#Compute Marginal likelihood
marginal_lik_2 = cumsum(trace_nstates_saved2.diagnostics[chain].ℓincrement for chain in eachindex(trace_nstates_saved2.diagnostics))
marginal_lik_3 = cumsum(trace_nstates_saved3.diagnostics[chain].ℓincrement for chain in eachindex(trace_nstates_saved3.diagnostics))
marginal_lik_5 = cumsum(trace_nstates_saved5.diagnostics[chain].ℓincrement for chain in eachindex(trace_nstates_saved5.diagnostics))
marginal_lik = [marginal_lik_2, marginal_lik_3, marginal_lik_5]

_fontsize3 = 20
_axissize3 = 20

#Plot incremental likelihoods
plot_score = plot(layout=(2,1), #plot(layout=(4,1),
    size = plot_default_size,
    legend=false,
    foreground_color_legend = :transparent,
    background_color_legend = :transparent,
    xguidefontsize=_fontsize3, yguidefontsize=_fontsize3, legendfontsize=_fontsize3,
    xtickfontsize=_axissize3, ytickfontsize=_axissize3
)
_param_color = :nipy_spectral
Nmodels = length(marginal_lik)
palette = Plots.palette(_param_color, 5)

############## MARGINAL LIKELIHOOD -> write it Cumulative Log PL
for iter in 1:2 #eachindex(marginal_lik)
    Plots.plot!(marginal_lik[iter], label= allnames[iter], legend=:topleft,
                ylabel="Cum. Log PL", color = palette[iter], subplot=1)
end
Plots.plot!(marginal_lik[3], label= allnames[3], legend=:topleft, line = :dash, linewidth=3,
            ylabel="Cum. Log PL", color = "gold4", subplot=1
)
#=
############## Bayes Factor -> Cumulative Log predictive Bayes Factor
log_bayes = [marginal_lik[2] .- marginal_lik[iter] for iter in eachindex(marginal_lik)]
for iter in eachindex(marginal_lik)
    Plots.plot!(log_bayes[iter], label= allnames[iter], legend=:topleft,
                ylabel="Log Bayes Factor - 3 state HSMM vs:", color = palette[iter], subplot=2)
end
=#
############## Data
Plots.plot!(f_nstates_data, label = false, ylabel="Data", xlabel="Time",
            legend=:topleft,color="black", subplot=2
)
plot_score
Plots.savefig("Chp6_DetermineStates_MarginalLik.pdf")

#Check summary of PMCMC 3 and 5
transform_nstates_3 = Baytes.TraceTransform(trace_nstates_saved3, model_nstates_3, Tagged(model_nstates_3, (:μ, :σ, :λ, :p) ),
    TransformInfo(collect(1:trace_nstates_saved3.info.sampling.Nchains), [1], 1:1:trace_nstates_saved3.info.sampling.iterations)
)
summary(trace_nstates_saved3, algorithm_nstates_3, transform_nstates_3)
chainsummary(trace_nstates_saved3, transform_nstates_3, Val(:text) )

transform_nstates_5 = Baytes.TraceTransform(trace_nstates_saved5, model_nstates_5, Tagged(model_nstates_5, (:μ, :σ, :λ, :p) ),
    TransformInfo(collect(1:trace_nstates_saved5.info.sampling.Nchains), [1], 1:1:trace_nstates_saved5.info.sampling.iterations)
)
summary(trace_nstates_saved5, algorithm_nstates_5, transform_nstates_5)
chainsummary(trace_nstates_saved5, transform_nstates_5, Val(:text) )
chainsummary(trace_nstates_saved5, transform_nstates_5, Val(:latex), PrintDefault(; Ndigits=2, quantiles=[0.025, 0.25, 0.50, 0.75, 0.975]) )
################################################################################
# Chapter 6 -

#######################################
# Log Vix Model comparison
plot_data = plot(
    size = plot_default_size, foreground_color_legend = nothing, legend=false,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize, margin=10Plots.mm
)
plot!(dates_fin, data_fin, xlabel = "Time", ylabel = "VIX Index Log Scale", legend=false, xticks=(_datayears,_dataticks))

#######################################
# Plot output of specific model
f_model = jldopen(string(pwd(), "/output/Real - SMC ARHSMM3 - Trace.jld2"))
f_modelname = "Chp6_ARHSMM3"

trace_real_saved = read(f_model, "trace");
model_real_saved = read(f_model, "model");
algorithm_real_saved = read(f_model, "algorithm");
tagged_real_mcmc = algorithm_real_saved.particles.kernel[1].pmcmc.kernel.mcmc.tune.tagged
tagged_real_pf = algorithm_real_saved.particles.kernel[1].pmcmc.kernel.pf.tune.tagged

transform_real = Baytes.TraceTransform(trace_real_saved, model_real_saved, tagged_real_mcmc,
    TransformInfo(collect(1:trace_real_saved.summary.info.Nchains), [1], 1:1:trace_real_saved.summary.info.iterations)
)
chainsummary(trace_real_saved, transform_real)
printchainsummary(trace_real_saved, transform_real, Val(:latex), PrintDefault(; Ndigits = 2))

plotChain(trace_real_saved, tagged_real_mcmc;
    _xaxis = dates_fin,
    burnin=00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
plot!(xticks=(_datayears,_dataticks))
_fontsize2 = 14
plot!(xguidefontsize=_fontsize2, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize2
)
Plots.savefig(string(f_modelname, "_SMC_CHAIN.pdf"))

plotLatent(trace_real_saved, tagged_real_pf;
    _xaxis = dates_fin,
    data = data_fin,
    burnin = 100,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)
plot!(xticks=(_datayears,_dataticks))
Plots.savefig(string(f_modelname, "_SMC_LATENT.pdf"))

plotPosteriorPrediction(trace_real_saved.diagnostics, _SMC2();
    _xaxis = dates_fin,
    data = data_fin,
    burnin=000,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize,
    CIRegion = [.025, .975]
)
plot!(xticks=(_datayears,_dataticks))
_fontsize2 = 16
plot!(xguidefontsize=_fontsize2, yguidefontsize=_fontsize2, legendfontsize=_fontsize2,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize2
)
Plots.savefig(string(f_modelname, "_SMC_PREDICT.pdf"))

plotDiagnostics(trace_real_saved.diagnostics, algorithm_real_saved;
    _xaxis = dates_fin,
    plotsize = plot_default_size,
    fontsize = _fontsize,
    axissize = _axissize
)
plot!(xticks=(_datayears,_dataticks))
_fontsize2 = 12
plot!(xguidefontsize=_fontsize, yguidefontsize=_fontsize2, legendfontsize=_fontsize2,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize2
)
Plots.savefig(string(f_modelname, "_SMC_DIAGNOSTICS.png"))

#######################################
# Clustering
_N0 = 1

#Get most likely filtered state
transform_real = Baytes.TraceTransform(trace_real_saved, model_real_saved, tagged_real_pf,
    TransformInfo(collect(1:trace_real_saved.summary.info.Nchains), [1], _N0:1:trace_real_saved.summary.info.iterations)
)
val = Baytes.get_chainvals(trace_real_saved, transform_real)

## Calculate Posterior mean in current chain
Nparticles = length(trace_real_saved.val)
NIterations = length(trace_real_saved.diagnostics)
Ninitial = length(val[1][begin].latent)
Nlatent = length(val[1][1].latent[1])

#preallocate buffer:
posteriormeans = [ [zeros(Float64, Ninitial+iter-1) for iter in Base.OneTo(NIterations-_N0+1)] for _ in Base.OneTo(Nlatent)]

# Get all posterior means
for t in Base.OneTo(NIterations-_N0+1)
    # Get all values in current time
    val_temp = [val[Nparticle][t].latent for Nparticle in 1:Nparticles]
    for state in 1:Nlatent
        # Get dimension of latent trajectory
        latent = [getfield.(val_temp[Nparticle], state) for Nparticle in 1:Nparticles]
        # Compute mean for each trajectory
        posteriormeans[state][t] = vec( mean( reduce(hcat,latent), dims=2) )
    end
end
posteriormeans
plot(posteriormeans[1][end])
# Round posterior mean
rounded =  Int.( round.( posteriormeans[1][end], digits = 0 ) )
plot(data_fin)
plot!(rounded .+ 1.5)

# get changes in vix levels
changes = data_fin[2:end] - data_fin[1:end-1]
pushfirst!(changes, 0.0)
changes

# Plot Changes
plot_hist = plot(layout=(1,1), #plot(layout=(4,1),
    size = plot_default_size,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize
)
plot!( ylabel = "Frequency", xlabel = "Change in Log Vix")
histogram!( changes[ rounded .== 1], bins=50, label= string("state 1 - ", Int(length(changes[ rounded .== 1])), " counts"), color="black")
histogram!( changes[ rounded .== 2], bins=50 , alpha = .5, label=string("state 2 - ", Int(length(changes[ rounded .== 2])), " counts"), color="gold4",)
histogram!( changes[ rounded .== 3], bins=120 , alpha = .5, label=string("state 3 - ", Int(length(changes[ rounded .== 3])), " counts"), color="red",)
plot_hist
Plots.savefig("Chp6_Clustering.pdf")


#Plot original time series
plot_cluster = plot(layout=(1,1), #plot(layout=(4,1),
    size = plot_default_size,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize
)
plot!(ylabel = "Log VIX", xlabel = "Time")
plot!(dates_fin, data_fin, label="Log VIX", color="black")

## Calculate Posterior mean accross chains
θᵐᵉᵃⁿ = posteriormeans[1][end]
#round to most likely state:
states_rescaled = round.(θᵐᵉᵃⁿ)
#Add most probable state
plot!(
    dates_fin, states_rescaled .- mean(states_rescaled) .+ mean(data_fin); #states_obs_scaled;
    lw=1.0,
    color="gold4",
    label="Most probable latent state, rescaled to fit data",
)
Plots.savefig("Chp6_ClusterOverTime.pdf")

#=
## Calculate Posterior mean accross chains
θᵐᵉᵃⁿ = posteriormeans[1][end]
states_uniform = (θᵐᵉᵃⁿ .- minimum(θᵐᵉᵃⁿ)) ./ (maximum(θᵐᵉᵃⁿ) - minimum(θᵐᵉᵃⁿ))
states_obs_scaled = states_uniform .* (maximum(data_fin) - minimum(data_fin)) #.+ mean( data_fin )
states_obs_scaled = states_obs_scaled .- mean(states_obs_scaled) .+ mean(data_fin)

states_uniform = states_uniform .- mean(states_uniform) .+ mean(data_fin)
plot!(
    dates_fin, states_uniform; #states_obs_scaled;
    lw=1.0,
    color="gold4",
    label="Rescaled State Posterior Mean at final iteration",
)
Plots.savefig("Chp6_ClusterOverTime.pdf")
=#
plot(plot_hist, plot_cluster, layout = (2,1), size=(1000,1000),
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_fontsize, ytickfontsize=_fontsize
)
plot!(xticks=(_datayears,_dataticks), subplot=2)
Plots.savefig("Chp6_ClusterSummary.pdf")

################################################################################
# Now compare this with HMM output
plot_hist_hsmm = deepcopy(plot_hist)

f_model = jldopen(string(pwd(), "/output/Real - SMC ARHMM3 - Trace.jld2"))
f_modelname = "Chp6_ARHMM3"

trace_real_saved = read(f_model, "trace");
model_real_saved = read(f_model, "model");
algorithm_real_saved = read(f_model, "algorithm");
tagged_real_mcmc = algorithm_real_saved.particles.kernel[1].pmcmc.kernel.mcmc.tune.tagged
tagged_real_pf = algorithm_real_saved.particles.kernel[1].pmcmc.kernel.pf.tune.tagged

transform_real = Baytes.TraceTransform(trace_real_saved, model_real_saved, tagged_real_mcmc,
    TransformInfo(collect(1:trace_real_saved.summary.info.Nchains), [1], 1:1:trace_real_saved.summary.info.iterations)
)
chainsummary(trace_real_saved, transform_real)
printchainsummary(trace_real_saved, transform_real, Val(:latex), PrintDefault(; Ndigits = 2))

plotChain(trace_real_saved, tagged_real_mcmc;
    _xaxis = dates_fin,
    burnin=00,
    plotsize = plot_default_size,
    param_color = plot_default_color,
    fontsize = _fontsize,
    axissize = _axissize
)

#######################################
# Clustering
#Get most likely filtered state
transform_real = Baytes.TraceTransform(trace_real_saved, model_real_saved, tagged_real_pf,
    TransformInfo(collect(1:trace_real_saved.summary.info.Nchains), [1], _N0:1:trace_real_saved.summary.info.iterations)
)
val = Baytes.get_chainvals(trace_real_saved, transform_real)

## Calculate Posterior mean in current chain
Nparticles = length(trace_real_saved.val)
NIterations = length(trace_real_saved.diagnostics)
Ninitial = length(val[1][begin].latent)
Nlatent = length(val[1][1].latent[1])

#preallocate buffer:
posteriormeans = [ [zeros(Float64, Ninitial+iter-1) for iter in Base.OneTo(NIterations-_N0+1)] for _ in Base.OneTo(Nlatent)]
# Get all posterior means
for t in Base.OneTo(NIterations-_N0+1)
    # Get all values in current time
    val_temp = [val[Nparticle][t].latent for Nparticle in 1:Nparticles]
    for state in 1:Nlatent
        # Get dimension of latent trajectory
#        latent = [getfield.(val_temp[Nparticle], state) for Nparticle in 1:Nparticles]
        latent = [val_temp[Nparticle] for Nparticle in 1:Nparticles]
        # Compute mean for each trajectory
        posteriormeans[state][t] = vec( mean( reduce(hcat,latent), dims=2) )
    end
end
posteriormeans
plot(posteriormeans[1][end])
# Round posterior mean
rounded =  Int.( round.( posteriormeans[1][end], digits = 0 ) )
plot(data_fin)
plot!(rounded .+ 1.5)

# get changes in vix levels
changes = data_fin[2:end] - data_fin[1:end-1]
pushfirst!(changes, 0.0)
changes

# Plot Changes
plot_hist = plot(layout=(1,1), #plot(layout=(4,1),
    size = plot_default_size,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize
)
plot!( ylabel = "Frequency", xlabel = "Change in Log Vix")
histogram!( changes[ rounded .== 1], bins=30, label= string("state 1 - ", Int(length(changes[ rounded .== 1])), " counts"), color="black")
histogram!( changes[ rounded .== 2], bins=100 , alpha = .5, label=string("state 2 - ", Int(length(changes[ rounded .== 2])), " counts"), color="gold4",)
histogram!( changes[ rounded .== 3], bins=30 , alpha = .5, label=string("state 3 - ", Int(length(changes[ rounded .== 3])), " counts"), color="red",)
plot_hist
#Plots.savefig("Chp6_Clustering.pdf")

plot_hist_hmm = deepcopy(plot_hist)

plot(plot_hist_hsmm, plot_hist_hmm)
xaxis!("AR HSMM - Change in Log VIX", subplot=1)
xaxis!("AR HMM - Change in Log VIX", subplot=2)
Plots.savefig("Chp6_ClusterComparison.pdf")

################################################################################
################################################################################
################################################################################
# Comparison of different models
algorithmnames = [
    "AR(1) HSMM, 2 state NegBin duration",
    "AR(1) HSMM, 3 state NegBin duration",
    "AR(1) HSMM, 4 state NegBin duration",

    "AR(1) HSMM, 2 state Poisson duration",
    "AR(1) HSMM, 3 state Poisson duration",
    "AR(1) HSMM, 4 state Poisson duration",

    "AR(1) HMM, 2 states",
    "AR(1) HMM, 3 states",
    "AR(1) HMM, 4 states",

    "AR(1)",
]
f_arhsmmNB2 = jldopen(string(pwd(), "/output/Real - SMC ARHSMM2 - Trace.jld2"));
f_arhsmmNB3 = jldopen(string(pwd(), "/output/Real - SMC ARHSMM3 - Trace.jld2"));
f_arhsmmNB4 = jldopen(string(pwd(), "/output/Real - SMC ARHSMM4 - Trace.jld2"));

f_arhsmmP2 = jldopen(string(pwd(), "/output/Real - SMC ARHSMMP2 - Trace.jld2"));
f_arhsmmP3 = jldopen(string(pwd(), "/output/Real - SMC ARHSMMP3 - Trace.jld2"));
f_arhsmmP4 = jldopen(string(pwd(), "/output/Real - SMC ARHSMMP4 - Trace.jld2"));

f_arhmm2 = jldopen(string(pwd(), "/output/Real - SMC ARHMM2 - Trace.jld2"));
f_arhmm3 = jldopen(string(pwd(), "/output/Real - SMC ARHMM3 - Trace.jld2"));
f_arhmm4 = jldopen(string(pwd(), "/output/Real - SMC ARHMM4 - Trace.jld2"));

f_ar1 = jldopen(string(pwd(), "/output/Real - SMC AR1 - Trace.jld2"));

SMC2_outputdiagnostics = [
    f_arhsmmNB2,
    f_arhsmmNB3,
    f_arhsmmNB4,

    f_arhsmmP2,
    f_arhsmmP3,
    f_arhsmmP4,

    f_arhmm2,
    f_arhmm3,
    f_arhmm4,
]
IBIS_outputdiagnostics = [f_ar1]
@argcheck length(algorithmnames) == length(SMC2_outputdiagnostics) + length(IBIS_outputdiagnostics)

cumsum_escores = []
cumsum_logscore_weighted = []
cumsum_logscore_unweighted = []
marginal_lik = []
_burnin = 0
data = data_fin

## SMC2
for idx in eachindex(SMC2_outputdiagnostics)
    println("Computing index ", idx, " ", algorithmnames[idx])
## Assign Variables
    trace_temp = read(SMC2_outputdiagnostics[idx], "trace")
    #model_temp = read(SMC2_outputdiagnostics[idx], "model")
    #algorithm_temp = read(SMC2_outputdiagnostics[idx], "algorithm")
## Add escore
    predictions_temp = [trace_temp.diagnostics[chain].base.prediction for chain in eachindex(trace_temp.diagnostics)]
    escore_temp = compute_escore([getfield.(predictions_temp[iter], 2) for iter in eachindex(predictions_temp)], data, _burnin)
    push!(cumsum_escores, cumsum(escore_temp))
## Add marginal likelihood
    push!(marginal_lik, cumsum([trace_temp.diagnostics[chain].ℓincrement for chain in eachindex(trace_temp.diagnostics)]))
## Add log score
    ℓweightsₙ_temp   = [trace_temp.diagnostics[iter].ℓweightsₙ for iter in eachindex(trace_temp.diagnostics)]
    logscore_weighted_temp, logscore_unweighted_temp = compute_ℓscore([getfield.(predictions_temp[iter], 2) for iter in eachindex(predictions_temp) ], ℓweightsₙ_temp, data)
    push!(cumsum_logscore_weighted, cumsum(logscore_weighted_temp))
    push!(cumsum_logscore_unweighted, cumsum(logscore_unweighted_temp))
end
## IBIS
for idx in eachindex(IBIS_outputdiagnostics)
## Assign Variables
    trace_temp = read(IBIS_outputdiagnostics[idx], "trace")
    #model_temp = read(SMC2_outputdiagnostics[idx], "model")
    #algorithm_temp = read(SMC2_outputdiagnostics[idx], "algorithm")
## Add escore
    predictions_temp = [trace_temp.diagnostics[chain].base.prediction for chain in eachindex(trace_temp.diagnostics)]
    escore_temp = compute_escore([predictions_temp[iter] for iter in eachindex(predictions_temp)], data, _burnin)
    push!(cumsum_escores, cumsum(escore_temp))
## Add marginal likelihood
    push!(marginal_lik, cumsum([trace_temp.diagnostics[chain].ℓincrement for chain in eachindex(trace_temp.diagnostics)]))
## Add log score
    ℓweightsₙ_temp   = [trace_temp.diagnostics[iter].ℓweightsₙ for iter in eachindex(trace_temp.diagnostics)]
    logscore_weighted_temp, logscore_unweighted_temp = compute_ℓscore([predictions_temp[iter] for iter in eachindex(predictions_temp) ], ℓweightsₙ_temp, data)
    push!(cumsum_logscore_weighted, cumsum(logscore_weighted_temp))
    push!(cumsum_logscore_unweighted, cumsum(logscore_unweighted_temp))
end

##########################################################################################
############## PLOTS
date_start = 1
date_end = length(marginal_lik[begin])

plot_score = plot(layout=(2,1), #plot(layout=(4,1),
    size = plot_default_size,
    legend=false,
    foreground_color_legend = :transparent,
    background_color_legend = :transparent,
    xguidefontsize=_fontsize, yguidefontsize=_fontsize, legendfontsize=_fontsize,
    xtickfontsize=_axissize, ytickfontsize=_axissize
)
_param_color = :nipy_spectral
Nmodels = length(algorithmnames)
palette = Plots.palette(_param_color, Nmodels+1)
alldata = data[(end-length(marginal_lik[begin])+1):end]
alldata = data[(end-length(marginal_lik[begin])+1):end]
alldates = dates_fin[(end-length(marginal_lik[begin])+1):end]

############## MARGINAL LIKELIHOOD
#=
for iter in eachindex(marginal_lik)
    Plots.plot!(marginal_lik[iter][date_start:date_end], label= algorithmnames[iter], legend=:topleft,
                ylabel="Marginal likelihood", color = palette[iter], subplot=1)
end
=#
############## Bayes Factor
_BenchmarkModel = 2
log_bayes = [marginal_lik[_BenchmarkModel] .- marginal_lik[iter] for iter in eachindex(marginal_lik)]
log_bayes
[marginal_lik[iter][end] for iter in eachindex(marginal_lik) ]
[log_bayes[iter][end] for iter in eachindex(log_bayes) ]

for iter in eachindex(marginal_lik)
    _logbayes = log_bayes[iter][date_start:date_end]
    Plots.plot!(alldates[date_start:date_end], _logbayes, label= algorithmnames[iter], legend=:topleft,
#                ylabel="Cum. Log Pred. BF \nof Winner", color = palette[iter], subplot=1)
                ylabel="CLPBF", color = palette[iter], subplot=1)
end
ylabel!( string("CLBF of \n", algorithmnames[_BenchmarkModel]), subplot=1)
plot_score

#=
############## ESCORE
for iter in eachindex(cumsum_escores)
    Plots.plot!(cumsum_escores[iter][date_start:date_end], label= algorithmnames[iter], legend=:topleft,
                ylabel="Cumulative Energy-score", palette = Plots.palette(_param_color, Nmodels), subplot=1)
end
=#

############## SAVE PLOT
_ndata = length(alldata[date_start:date_end])
Plots.plot!(alldates[date_start:date_end], alldata[date_start:date_end], label = false, ylabel="Data after Initialization", xlabel="Time",
            legend=:topleft,color="black", subplot=2
)
plot!(xticks=(_datayears,_dataticks))
Plots.savefig("Chp6_BayesFactor.pdf")

########################################################################
# TABLES
_idx = date_start:date_end

_escores = map(iter -> cumsum_escores[iter][date_end], eachindex(cumsum_escores))
_marginal_lik = map(iter -> marginal_lik[iter][date_end], eachindex(marginal_lik))
_logscore_weighted = map(iter -> cumsum_logscore_weighted[iter][date_end], eachindex(cumsum_logscore_weighted))
_logscore_unweighted = map(iter -> cumsum_logscore_unweighted[iter][date_end], eachindex(cumsum_logscore_unweighted))

chain_table = reduce(hcat, [algorithmnames, _escores, _marginal_lik])
using PrettyTables
println("Data points from index ", date_start, " until index ", date_end, ":")
pretty_table(chain_table,
#    backend = Val(:latex),
    #header = ["Names"," Cum. CRPS", "Last Marginal Log Likelihood", "Cum LogScore Weighted", "Cum LogScore Unweighted"],  #Parameter
    header = ["Names"," Cum. CRPS", "Last Marginal Log Likelihood"]#,  #Parameter
#    crop = :none
)

#Relative table from marginal likelihood
_marginal_lik = map(iter -> marginal_lik[iter][date_end], eachindex(marginal_lik))

# Marginal lik differential against winning model
_winner = 2
marginal_lik_diff = [_marginal_lik[_winner] - _marginal_lik[iter] for iter in eachindex(_marginal_lik)]

chain_table = reduce(hcat, [algorithmnames, round.(_marginal_lik; digits=2), round.(marginal_lik_diff; digits=2)])

using PrettyTables
println("Data points from index ", date_start, " until index ", date_end, ":")
pretty_table(chain_table,
    backend = Val(:latex),
    #header = ["Names"," Cum. CRPS", "Last Marginal Log Likelihood", "Cum LogScore Weighted", "Cum LogScore Unweighted"],  #Parameter
    header = ["Names"," Cum Log PL.", "Difference to Winner"]#,  #Parameter
#    crop = :none
)
