latent_init = rand(_rng, Categorical(2), n)
################################################################################
# AR HSMM
#2 state case
param_arhmm2 = (   μ = Param([3.0, 1.0],
                    [truncated(Normal(3.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)]),
                    σ = Param([.2, .4],
                    [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)]),
                    w = Param([.2, .8],
                    [truncated(Normal(0.0,10^5.), -.5, .5), truncated(Normal(0.8,10^5.), 0.0, 1.)]),
                    p = Param([[.95, .05], [.7, .3]], [Dirichlet(2,2) for i in 1:2]),
                   latent = Param(latent_init, Fixed())
                )
struct ARHMM2 <: ModelName end
arhmm2 = ModelWrapper(ARHMM2(), param_arhmm2)

function get_dynamics(model::ModelWrapper{<:ARHMM2}, θ)
    @unpack p = θ
    dynamicsˢ = [ Categorical( p[iter] ) for iter in eachindex(p) ]
    return dynamicsˢ
end
################################################################################
# Define sample and likelihood
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{F}; Nsamples = 1000)  where {F<:ARHMM2}
    ## Create distributions
    @unpack μ, σ, w = model.val
    dynamicsˢ = get_dynamics(model, model.val)
## Assign data container
    latentⁿᵉʷ = zeros(Int64, Nsamples+1)
    observedⁿᵉʷ = initzeros(0., Nsamples+1)
## Sample initial state and observation
    stateₜ = rand( dynamicsˢ[ rand(1:length(dynamicsˢ) ) ] )
    latentⁿᵉʷ[1] = stateₜ
    observedⁿᵉʷ[1] = rand(Normal(μ[stateₜ], σ[stateₜ]))
## Propagate initial observation and latent state forward until it reached data memory
    for iter in 2:length(observedⁿᵉʷ)
        stateₜ = rand(dynamicsˢ[stateₜ])
        fillvec!(latentⁿᵉʷ, stateₜ, iter)
        μᵗ = μ[stateₜ] + w[stateₜ]*observedⁿᵉʷ[iter-1]
        observedⁿᵉʷ[iter] = rand(Normal(μᵗ, σ[stateₜ]))
    end
## Return all data after initial draw
    return observedⁿᵉʷ[2:end], latentⁿᵉʷ[2:end]
end

function (objective::Objective{<:ModelWrapper{M}})(θ::NamedTuple) where {M<:ARHMM2}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.constraint, BaytesCore.subset(θ, tagged.parameter))
## Likelihood
    @unpack μ, σ, w, latent = θ
    dynamicsˢ = get_dynamics(model, θ)
    ll = 0.0
    structure = ByRows()
    for iter in 2:size(data,1)
        ##Update state
        sₜ₋₁  = grab(latent, iter-1, structure)
        sₜ = grab(latent, iter, structure)
        μᵗ = μ[sₜ] + w[sₜ]*data[iter-1]
        ll += logpdf(Normal(μᵗ, σ[sₜ]), data[iter])
        ll += logpdf(dynamicsˢ[sₜ₋₁], sₜ)
    end
    return ll + lp
end
dat, lat = simulate(_rng, arhmm2)
_obj = Objective(arhmm2, dat,)
_obj(_obj.model.val)

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:ARHMM2}
    @unpack model, data, tagged = objective
    @unpack μ, σ, w, p, latent = model.val
    # get new latent variable
    latent_new = rand(_rng, Categorical(p[latent[end]]))
    μᵗ = μ[latent_new] + w[latent_new]*data[end]
    return rand(_rng, Normal(μᵗ, σ[latent_new]) )
end

data_arhmm2, latent_arhmm2 = simulate(_rng, arhmm2; Nsamples = n)
_tagged_arhmm2 = Tagged(arhmm2, :latent)
fill!(arhmm2, _tagged_arhmm2, (; latent = latent_arhmm2))

################################################################################
#Assign dynamics

#Univariate
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{M}}) where {M<:ARHMM2}
    @unpack model, data = objective
    dynamicsˢ = get_dynamics(model, model.val)
    @unpack w, μ, σ = model.val

    initialˢ =  Categorical( get_stationary( reduce(hcat, [dynamicsˢ[iter].p for iter in eachindex(dynamicsˢ)] )' ) )
    state(particles, iter) = dynamicsˢ[particles[iter-1]]

    observation(particles, iter) = Normal(μ[particles[iter]] + w[particles[iter]]*data[iter-1], σ[particles[iter]])
    return Markov(initialˢ, state, observation)
end
