################################################################################
#struct ARHSMM2 <: ModelName end
#struct ARHSMMP2 <: ModelName end
struct ARHSMM <: ModelName end
struct ARHSMMP <: ModelName end

################################################################################
# AR HSMM
latent_init = [(rand(_rng, Categorical(2)), rand(_rng, Poisson(10) ) ) for iter in 1:n]

#2 state case
param_arhsmm2 = (
    μ = Param(
        [truncated(Normal(3.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)],
        [3.0, 1.0],
    ),
    σ = Param(
        [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)],
        [.2, .4],
    ),
    w = Param(
        [truncated(Normal(0.0,10^5.), -.5, .5), truncated(Normal(0.8,10^5.), 0.0, 1.)],
        [.2, .8],
    ),
    p = Param(
        Fixed(),
        [[1], [1]],
    ),
    r = Param(
        [truncated(Normal(10.0, 10^5), 0.0, 50.0), truncated(Normal(5.0, 10^5), 0.0, 20.0)],
        [5., 2.5],
    ),
    ϕ = Param(
        [Beta(1., 1.), Beta(1., 1.)],
        [.1, .1],
    ),
    latent = Param(
        Fixed(),
        latent_init,
    )
)
param_arhsmm3 = (
    μ = Param(
        [truncated(Normal(0.2,10^5.), 0.01, 1.0), truncated(Normal(0.2,10^5.), 0.01, 0.3), truncated(Normal(0.2,10^5.), 0.01, 1.0)],
        [0.1, 0.2, 0.3],
    ),
    σ = Param(
        [truncated(Normal(0.2,10^5.), 0.01, 1.0), truncated(Normal(0.2,10^5.), 0.01, 0.3), truncated(Normal(0.2,10^5.), 0.01, 0.3)],
        [0.1, 0.2, 0.3],#        [.2, .1],
    ),
    w = Param(
        [truncated(Normal(0.1,10^5.), 0.0, 1.0), truncated(Normal(0.1,10^5.), 0.0, 1.0), truncated(Normal(0.1,10^5.), 0.0, 1.0)],
        [.8, .9, 0.9],
    ),
    p = Param(
        [Dirichlet(2,2), Dirichlet(2,2), Dirichlet(2,2)],
        [[.5, .5], [.5, .5], [.5, .5]],
        ),
    r = Param(
        [truncated(Normal(10.0, 10^5), 0.0, 20.0), truncated(Normal(10.0, 10^5), 0.0, 20.0), truncated(Normal(10.0, 10^5), 0.0, 20.0)],
        [5.0, 5.0, 5.0],
    ),
    ϕ = Param(
        [Beta(1., 1.), Beta(1., 1.), Beta(1., 1.)],
        [0.1, 0.1, 0.1],
    ),
    latent = Param(
        Fixed(),
        latent_init,
    )
)

param_arhsmmP2 = (
    μ = Param(
        [truncated(Normal(3.0,10^5.), 0., 10.), truncated(Normal(1.0,10^5.), 0., 10.)],
        [3.0, 1.0],
    ),
    σ = Param(
        [truncated(Normal(0.2,10^5.), 0.01, 5.), truncated(Normal(0.4,10^5.), 0.01, 5.)],
        [.2, .4],
    ),
    w = Param(
        [truncated(Normal(0.0,10^5.), -.5, .5), truncated(Normal(0.8,10^5.), 0.0, 1.)],
        [.2, .8],
    ),
    p = Param(
        Fixed(),
        [[1], [1]],
    ),
    λ = Param(
        [truncated(Normal(30.0, 10^5), 0.0, 100.0), truncated(Normal(10.0, 10^5), 0.0, 30.0)],
        [30., 10.],
    ),
    latent = Param(
        Fixed(),
        latent_init,
    )
)
param_arhsmmP3 = (
    μ = Param(
        [truncated(Normal(0.2,10^5.), 0.01, 1.0), truncated(Normal(0.2,10^5.), 0.01, 0.3), truncated(Normal(0.2,10^5.), 0.01, 1.0)],
        [0.1, 0.2, 0.3],
    ),
    σ = Param(
        [truncated(Normal(0.2,10^5.), 0.01, 1.0), truncated(Normal(0.2,10^5.), 0.01, 0.3), truncated(Normal(0.2,10^5.), 0.01, 0.3)],
        [0.1, 0.2, 0.3],#        [.2, .1],
    ),
    w = Param(
        [truncated(Normal(0.1,10^5.), 0.0, 1.0), truncated(Normal(0.1,10^5.), 0.0, 1.0), truncated(Normal(0.1,10^5.), 0.0, 1.0)],
        [.8, .9, 0.9],
    ),
    p = Param(
        [Dirichlet(2,2), Dirichlet(2,2), Dirichlet(2,2)],
        [[.5, .5], [.5, .5], [.5, .5]],
        ),
    λ = Param(
        [truncated(Normal(30.0, 10^5), 0.0, 100.0), truncated(Normal(10.0, 10^5), 0.0, 30.0), truncated(Normal(10.0, 10^5), 0.0, 30.0)],
        [30., 10., 10.0],
    ),
    latent = Param(
        Fixed(),
        latent_init,
    )
)

function get_dynamics(model::ModelWrapper{A}, θ) where {A<:ARHSMM}
    @unpack μ, σ, r, ϕ, p = θ
    dynamicsᵈ = [ NegativeBinomial(r[iter], ϕ[iter]) for iter in eachindex(μ) ]
    dynamicsˢ = [ Categorical( extend_state(p[iter], iter) ) for iter in eachindex(μ) ]
    return dynamicsˢ, dynamicsᵈ
end
function get_dynamics(model::ModelWrapper{M}, θ) where {M<:ARHSMMP}
    @unpack μ, σ, λ, p = θ
    dynamicsᵈ = [ Poisson(λ[iter]) for iter in eachindex(μ) ]
    dynamicsˢ = [ Categorical( extend_state(p[iter], iter) ) for iter in eachindex(μ) ]
    return dynamicsˢ, dynamicsᵈ
end

################################################################################
# Define sample and likelihood
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{F}; Nsamples = 1000)  where {F<:Union{ARHSMM, ARHSMMP}}
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

function (objective::Objective{<:ModelWrapper{M}})(θ::NamedTuple) where {M<:Union{ARHSMM, ARHSMMP}}
    @unpack model, data, tagged = objective
## Prior
    lp = log_prior(tagged.info.transform.constraint, BaytesCore.subset(θ, tagged.parameter))
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

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Union{ARHSMM, ARHSMMP}}
    @unpack model, data, tagged = objective
    @unpack μ, σ, w, p, latent = model.val
    # get new latent variable
    if latent[end][2] != 0
        latent_new = latent[end][1]
    else
        _state = latent[end][1]
        latent_new = rand(_rng, Categorical( extend_state(p[_state], _state) ) )
    end
    μᵗ = μ[latent_new] + w[latent_new]*data[end]
    return rand(_rng, Normal(μᵗ, σ[latent_new]) )
end
################################################################################
#Assign dynamics

#Univariate
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{M}}) where {M<:Union{ARHSMM, ARHSMMP}}
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

################################################################################
arhsmm2 = ModelWrapper(ARHSMM(), param_arhsmm2)
arhsmmP2 = ModelWrapper(ARHSMMP(), param_arhsmmP2)

tagged_arhsmm2 = Tagged(arhsmm2, (:μ, :σ, :w, :r, :ϕ) )
tagged_arhsmmP2 = Tagged(arhsmmP2, (:μ, :σ, :w, :λ) )

data_arhsmm2, latent_arhsmm2 = simulate(_rng, arhsmm2; Nsamples = n)
_tagged_arhsmm2 = Tagged(arhsmm2, :latent)
fill!(arhsmm2, _tagged_arhsmm2, (; latent = latent_arhsmm2))

data_arhsmmP2, latent_arhsmmP2 = simulate(_rng, arhsmmP2; Nsamples = n)
_tagged_arhsmmP2 = Tagged(arhsmmP2, :latent)
fill!(arhsmmP2, _tagged_arhsmmP2, (; latent = latent_arhsmmP2))

_obj = Objective(arhsmm2, data_arhsmm2)
_obj(_obj.model.val)
dynamics(_obj)

_objP = Objective(arhsmmP2, data_arhsmmP2)
_objP(_objP.model.val)
dynamics(_objP)

arhsmm3 = ModelWrapper(ARHSMM(), param_arhsmm3)
arhsmmP3 = ModelWrapper(ARHSMMP(), param_arhsmmP3)

tagged_arhsmm3 = Tagged(arhsmm3, (:μ, :σ, :w, :p, :r, :ϕ) )
tagged_arhsmmP3 = Tagged(arhsmmP3, (:μ, :σ, :w, :p, :λ) )

data_arhsmm3, latent_arhsmm3 = simulate(_rng, arhsmm3; Nsamples = n)
_tagged_arhsmm3 = Tagged(arhsmm3, :latent)
fill!(arhsmm3, _tagged_arhsmm3, (; latent = latent_arhsmm3))

data_arhsmmP3, latent_arhsmmP3 = simulate(_rng, arhsmmP3; Nsamples = n)
_tagged_arhsmmP3 = Tagged(arhsmmP3, :latent)
fill!(arhsmmP3, _tagged_arhsmmP3, (; latent = latent_arhsmmP3))

_obj = Objective(arhsmm3, data_arhsmm3)
_obj(_obj.model.val)
dynamics(_obj)

_objP = Objective(arhsmmP3, data_arhsmmP3)
_objP(_objP.model.val)
dynamics(_objP)
