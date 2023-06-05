################################################################################
struct MarkovJump <: ModelName end

################################################################################
# Markov Jump Model
latent_init = rand(_rng, Bernoulli(0.1), n)
param_MarkovJump = (;
    μ = Param(
        truncated(Normal(0., 10^5), -10., 10.0),
        0.1,
    ),
    σ = Param(
        truncated(Normal(.5, 10^5), 0., 10.0),
        0.5,
    ),
    μⱼ = Param(
        truncated(Normal(0., 10^5), -10., 10.0),
        -1.0,
    ),
    σⱼ = Param(
        truncated(Normal(.5, 10^5), 0., 10.0),
        1.0,
    ),
    λ = Param(
        Beta(1., 4.),
        0.1,
    ),
    latent = Param(
        Fixed(),
        rand( Bernoulli(0.1), n),
    ),
)

################################################################################
# Define sample and likelihood
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{<:MarkovJump}; Nsamples = 1000)
    @unpack μ, σ, μⱼ, σⱼ, λ = model.val
    ## Create distributions
    dynamicsˢ = Bernoulli( λ )
## Assign data container
    latentⁿᵉʷ   = initzeros( rand(dynamicsˢ), Nsamples)
    observedⁿᵉʷ = initzeros( rand( Normal(μ, σ) ) , Nsamples)

    stateₜ  = rand( dynamicsˢ )
    eₜ      = rand( Normal(μ + stateₜ*μⱼ, σ + stateₜ*σⱼ) )
    fillvec!(latentⁿᵉʷ, stateₜ, 1 )
    fillvec!(observedⁿᵉʷ, eₜ, 1 )

## Now propagate Particles forward together
    for iter in 2:size( observedⁿᵉʷ,1 )
            stateₜ  = rand( dynamicsˢ ) #stateₜ for t-1 overwritten
            eₜ      = rand( Normal(μ + stateₜ*μⱼ, σ + stateₜ*σⱼ) )
            fillvec!(latentⁿᵉʷ, stateₜ, iter)
            fillvec!(observedⁿᵉʷ, eₜ, iter )
    end
## Return data
    return observedⁿᵉʷ, latentⁿᵉʷ
end

#=
!NOTE: Target: P(e₁_ₜ, s₁_ₜ, θ)
        = ∑ₜ P(eₜ | sₜ | θ)
=#
function (objective::Objective{<:ModelWrapper{<:MarkovJump}})(θ::NamedTuple)
    @unpack model, data, tagged = objective
    @unpack μ, σ, μⱼ, σⱼ, λ, latent = θ
## Prior
    lp = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    ll = 0.0
    ## Create distributions
    ll += sum( logpdf(Bernoulli(λ), latent[t]) for t in eachindex(latent)  )
    ll += sum( logpdf(Normal(μ + latent[t]*μⱼ, σ + latent[t]*σⱼ), data[t]) for t in eachindex(data) )
    return ll + lp
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{<:MarkovJump}})
    @unpack model, data = objective
    @unpack μ, σ, μⱼ, σⱼ, λ, latent = model.val
    #!NOTE: first predict latent then data
    latent_new = rand(_rng, Bernoulli(λ) )
    return rand(_rng, Normal(μ + latent_new*μⱼ, σ + latent_new*σⱼ) )
end

################################################################################
#Assign dynamics

#Univariate
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{<:MarkovJump}})
    @unpack μ, σ, μⱼ, σⱼ, λ = objective.model.val
    initial = Bernoulli(0.5)
    transition(particles, iter) = Bernoulli(λ)
    evidence(particles, iter) =  Normal(μ + particles[iter]*μⱼ, σ + particles[iter]*σⱼ)

    return Markov(initial, transition, evidence)
end

################################################################################
mj = ModelWrapper(MarkovJump(), param_MarkovJump)

tagged_mj = Tagged(mj, (:μ, :σ, :μⱼ, :σⱼ, :λ) )

data_mj, latent_mj = simulate(_rng, mj; Nsamples = n)
_tagged = Tagged(mj, :latent)

fill!(mj, _tagged, (; latent = latent_mj))
objectiveUV = Objective(mj, data_mj,  Tagged(mj, (:μ, :σ, :μⱼ, :σⱼ, :λ)))
objectiveUV(mj.val)
predict(_rng, objectiveUV)

BaytesFilters.dynamics(objectiveUV)

################################################################################
# Likelihood integrated out

#=
!NOTE: Target: log P(e₁_ₜ | θ)
        = ∑ₜ log ∑ₛₜₐₜₑ P(eₜ, sₜ = state | θ)
        = ∑ₜ log ∑ₛₜₐₜₑ P(eₜ | sₜ = state,  θ) P(sₜ = state | θ)
=#
import BaytesInference: filter_forward
function filter_forward(objective::Objective{<:ModelWrapper{<:MarkovJump}})
    @unpack model, data = objective
    @unpack μ, σ, μⱼ, σⱼ, λ = model.val
    ll = 0.0
    for t in eachindex(data)
        ll += log(sum(  Distributions.pdf(Bernoulli(λ), state) * Distributions.pdf( Normal(μ + state*μⱼ, σ + state*σⱼ), data[t]  ) for state in 0:1 ) )
    end
    return ll
end
