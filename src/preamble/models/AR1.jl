################################################################################
struct AR1 <: ModelName end

################################################################################
param_ar1 = (
    μ = Param(
        truncated(Normal(3.0,10^5.), 0., 10.),
        0.5,
    ),
    σ = Param(
        truncated(Normal(0.2,10^5.), 0.01, 5.),
        .2,
    ),
    w = Param(
        truncated(Normal(0.0,10^5.), .0, 1.0),
        .8,
    ),
)

function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{F}; Nsamples = 1000)  where {F<:AR1}
    @unpack μ, σ, w = model.val
## Assign data container
    observedⁿᵉʷ = initzeros(0., Nsamples+1)
## Sample initial observation
    observedⁿᵉʷ[1] = rand(Normal(μ, σ))
## Propagate initial observation and latent state forward until it reached data memory
    for iter in 2:length(observedⁿᵉʷ)
        μᵗ = μ + w*observedⁿᵉʷ[iter-1]
        observedⁿᵉʷ[iter] = rand(Normal(μᵗ, σ))
    end
    return observedⁿᵉʷ[2:end]
end

function (objective::Objective{<:ModelWrapper{M}})(θ::NamedTuple) where {M<:AR1}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.transform.constraint, BaytesCore.subset(θ, tagged.parameter))
## Likelihood
    @unpack μ, σ, w = θ
    ll = 0.0
    structure = ByRows()
    for iter in 2:size(data,1)
        μᵗ = μ + w*data[iter-1]
        ll += logpdf(Normal(μᵗ, σ), data[iter])
    end
    return ll + lp
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:AR1}
    @unpack model, data, tagged = objective
    @unpack μ, σ, w = model.val
    μᵗ = μ + w*data[end]
    return rand(_rng, Normal(μᵗ, σ) )
end

function BaytesSMC.SMCweight(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}, algorithm, cumweightsₜ₋₁) where {M<:AR1}
    @unpack model, data, tagged, temperature = objective
## Likelihood
    @unpack μ, σ, w = model.val
    ll = 0.0
    structure = ByRows()
    for iter in 2:size(data,1)
        μᵗ = μ + w*data[iter-1]
        ll += logpdf(Normal(μᵗ, σ), data[iter])
    end
    cumweightsₜ = temperature*ll
    return cumweightsₜ, cumweightsₜ - cumweightsₜ₋₁
end

################################################################################
ar1 = ModelWrapper(AR1(), param_ar1)

tagged_ar1 = Tagged(ar1, (:μ, :σ, :w) )

dat = simulate(_rng, ar1)
_obj = Objective(ar1, dat,)
_obj(_obj.model.val)

predict(_rng, _obj)

BaytesSMC.SMCweight(_rng, _obj, 1., 2.)
