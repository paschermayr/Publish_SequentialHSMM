param_ar1 = (
    μ = Param(0.5, truncated(Normal(3.0,10^5.), 0., 10.)),
    σ = Param(.2, truncated(Normal(0.2,10^5.), 0.01, 5.)),
    w = Param(.8, truncated(Normal(0.0,10^5.), .0, 1.0)),
)

struct AR1 <: ModelName end
ar1 = ModelWrapper(AR1(), param_ar1)

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
    lp = log_prior(tagged.info.constraint, BaytesCore.subset(θ, tagged.parameter))
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
dat = simulate(_rng, ar1)
_obj = Objective(ar1, dat,)
_obj(_obj.model.val)

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:AR1}
    @unpack model, data, tagged = objective
    @unpack μ, σ, w = model.val
    μᵗ = μ + w*data[end]
    return rand(_rng, Normal(μᵗ, σ) )
end
predict(_rng, _obj)

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
BaytesSMC.SMCweight(_rng, _obj, 1., 2.)
