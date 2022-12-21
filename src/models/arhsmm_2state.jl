latent_init = [(rand(_rng, Categorical(2)), rand(_rng, Poisson(10) ) ) for iter in 1:n]
################################################################################
# AR HSMM
#2 state case
param_arhsmm2 = (   μ = Param([3.0, 1.0],
                    [truncated(Normal(3.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)]),
                    σ = Param([.2, .4],
                    [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)]),
                    w = Param([.2, .8],
                    [truncated(Normal(0.0,10^5.), -.5, .5), truncated(Normal(0.8,10^5.), 0.0, 1.)]),
                    r = Param([5., 2.5],
                        [truncated(Normal(10.0, 10^5), 0.0, 50.0), truncated(Normal(5.0, 10^5), 0.0, 20.0)]),
                    ϕ = Param([.1, .1],
                        [Beta(1., 1.), Beta(1., 1.)]),
                   latent = Param(latent_init, Fixed())
                )
struct ARHSMM2 <: ModelName end
arhsmm2 = ModelWrapper(ARHSMM2(), param_arhsmm2)

param_arhsmmP2 = (   μ = Param([3.0, 1.0],
                    [truncated(Normal(3.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)]),
                    σ = Param([.2, .4],
                    [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)]),
                    w = Param([.2, .8],
                    [truncated(Normal(0.0,10^5.), -.5, .5), truncated(Normal(0.8,10^5.), 0.0, 1.)]),
                    λ = Param([30., 10.],
                        [truncated(Normal(30.0, 10^5), 0.0, 100.0), truncated(Normal(10.0, 10^5), 0.0, 30.0)]),
                   latent = Param(latent_init, Fixed())
                )
struct ARHSMMP2 <: ModelName end
arhsmmP2 = ModelWrapper(ARHSMMP2(), param_arhsmmP2)

function get_dynamics(model::ModelWrapper{<:ARHSMM2}, θ)
    @unpack μ, σ, r, ϕ = θ
    dynamicsˢ = [Categorical(extend_state(1., iter)) for iter in eachindex(μ)]
    dynamicsᵈ = [ NegativeBinomial(r[iter], ϕ[iter]) for iter in eachindex(r) ]
    return dynamicsˢ, dynamicsᵈ
end
function get_dynamics(model::ModelWrapper{<:ARHSMMP2}, θ)
    @unpack μ, σ, λ = θ
    dynamicsˢ = [Categorical(extend_state(1., iter)) for iter in eachindex(μ)]
    dynamicsᵈ = [Poisson(λ[iter]) for iter in eachindex(λ) ]
    return dynamicsˢ, dynamicsᵈ
end
################################################################################
# Define sample and likelihood
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{F}; Nsamples = 1000)  where {F<:Union{ARHSMM2, ARHSMMP2}}
    ## Create distributions
    @unpack μ, σ, w = model.val
    dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)
## Assign data container
    latentⁿᵉʷ = [(0,0) for _ in Base.OneTo(Nsamples+1)]
    observedⁿᵉʷ = initzeros( 0., Nsamples+1)
## Sample initial state and observation
    stateₜ = rand( dynamicsˢ[ rand(1:length(dynamicsˢ) ) ] )
    durationₜ = rand( dynamicsᵈ[stateₜ])
    fillvec!(latentⁿᵉʷ, (stateₜ, durationₜ), 1 )
    observedⁿᵉʷ[1] = rand(Normal(μ[stateₜ], σ[stateₜ]))
## Propagate initial observation and latent state forward until it reached data memory
    for iter in 2:length(observedⁿᵉʷ)
        if durationₜ > 0
            durationₜ -=  1
            fillvec!(latentⁿᵉʷ, (stateₜ, durationₜ), iter )
        else
            stateₜ = rand( dynamicsˢ[ stateₜ ] ) #stateₜ for t-1 overwritten
            durationₜ = rand( dynamicsᵈ[stateₜ])
            fillvec!(latentⁿᵉʷ, (stateₜ, durationₜ), iter )
        end
        μᵗ = μ[stateₜ] + w[stateₜ]*observedⁿᵉʷ[iter-1]
        observedⁿᵉʷ[iter] = rand(Normal(μᵗ, σ[stateₜ]))
    end
## Return all data after initial draw
    return observedⁿᵉʷ[2:end], latentⁿᵉʷ[2:end]
end

function (objective::Objective{<:ModelWrapper{M}})(θ::NamedTuple) where {M<:Union{ARHSMM2, ARHSMMP2}}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.constraint, BaytesCore.subset(θ, tagged.parameter))
## Likelihood
    @unpack μ, σ, w, latent = θ
    dynamicsˢ, dynamicsᵈ = get_dynamics(model, θ)
    ll = 0.0
    structure = ByRows()
    for iter in 2:size(data,1)
        ##Update state
        sₜ₋₁, dₜ₋₁  = grab(latent, iter-1, structure)
        sₜ, dₜ   = grab(latent, iter, structure)
        μᵗ = μ[sₜ] + w[sₜ]*data[iter-1]
        ll += logpdf(Normal(μᵗ, σ[sₜ]), data[iter])
        if sₜ₋₁ != sₜ
            ll += logpdf(dynamicsˢ[sₜ₋₁], sₜ)
            ll += logpdf(dynamicsᵈ[sₜ], dₜ)
        end
    end
    return ll + lp
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Union{ARHSMM2, ARHSMMP2}}
    @unpack model, data, tagged = objective
    @unpack μ, σ, w, latent = model.val
    # get new latent variable
    if latent[end][2] != 0
        latent_new = latent[end][1]
    else
        latent_new = rand(_rng, Categorical( extend_state(1., latent[end][1]) ) )
    end
    μᵗ = μ[latent_new] + w[latent_new]*data[end]
    return rand(_rng, Normal(μᵗ, σ[latent_new]) )
end

data_arhsmm2, latent_arhsmm2 = simulate(_rng, arhsmm2; Nsamples = n)
_tagged_arhsmm2 = Tagged(arhsmm2, :latent)
fill!(arhsmm2, _tagged_arhsmm2, (; latent = latent_arhsmm2))

data_arhsmmP2, latent_arhsmmP2 = simulate(_rng, arhsmmP2; Nsamples = n)
_tagged_arhsmmP2 = Tagged(arhsmmP2, :latent)
fill!(arhsmmP2, _tagged_arhsmmP2, (; latent = latent_arhsmmP2))

################################################################################
#Assign dynamics

#Univariate
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{M}}) where {M<:Union{ARHSMM2, ARHSMMP2}}
    @unpack model, data = objective
    dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)
    @unpack w, μ, σ = model.val

    initialˢ = Categorical( get_stationary( reduce(hcat, [dynamicsˢ[iter].p for iter in eachindex(dynamicsˢ)] )' ) )
    initialᵈ(sₜ::R) where {R}               = dynamicsᵈ[sₜ]
    initial = SemiMarkovInitiation(initialˢ, initialᵈ)

    state(particles, iter) = dynamicsˢ[particles[iter-1][1]]
    duration(s, iter) = dynamicsᵈ[s]
    transition = SemiMarkovTransition(state, duration)

    observation(particles, iter) = Normal(μ[particles[iter][1]] + w[particles[iter][1]]*data[iter-1], σ[particles[iter][1]])
    return SemiMarkov(initial, transition, observation)
end
