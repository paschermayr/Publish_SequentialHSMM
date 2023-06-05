################################################################################
struct HSMM_UV_P <: ModelName end
struct HSMM_UV <: ModelName end
struct HSMM_UV3 <: ModelName end
struct HSMM_UV_P5 <: ModelName end

################################################################################
#UNIVARIATE CASE

latent_init = [( convert(latent_type, rand(Categorical(2)) ), convert(latent_type, rand(Poisson(10) ) ) ) for iter in 1:n]
latent_init5 = [( convert(latent_type, rand(Categorical(5)) ), convert(latent_type, rand(Poisson(10) ) ) ) for iter in 1:n]

#s_0 = [convert(latent_type, rand(Categorical(3))) for iter in 1:n]
#d_0 = [convert(latent_type, rand(Poisson(10))) for iter in 1:n]

param_HSMM_UV_P = (;
    μ = Param(
        [truncated(Normal(-2., 100.), -10., 0.0), truncated(Normal(2., 100.), 0.0, 10.0) ],
        [-2., 2.],
    ),
    σ = Param(
        [truncated(Normal(5.0, 20.), 0.0, 10.0), truncated(Normal(2.5, 20.), 0.0, 10.0)],
        [5., 2.],
    ),
    λ = Param(
        [truncated(Normal(10.0, 10^5), 0.0, 50.0), truncated(Normal(50.0, 10^5), 0.0, 100.0)],
        [10., 50.],
    ),
    p = Param(
        Fixed(),
        [[1], [1]],
    ),
    latent = Param(
        Fixed(),
        latent_init,
    ),
    )
param_HSMM_UV = (;
    μ = Param(
        [truncated(Normal(-2., 100,), -10., 0.0), truncated(Normal(2., 100.), 0.0, 10.0) ],
        [-2., 2.],
    ),
    σ = Param(
        [truncated(Normal(5., 20.), 0.0, 10.0), truncated(Normal(2.0, 20.), 0.0, 10.0)],
        [4., 2.],
    ),
    r = Param(
        [truncated(Normal(10.0, 10^5), 0.0, 20.0), truncated(Normal(20.0, 10^5), 0.0, 100.0)],
        [10., 15.],
    ),
    ϕ = Param(
        [Beta(1., 1.5), Beta(1., 1.5)],
        [.3, .3],
    ),
    p = Param(
        Fixed(),
        [[1], [1]],
    ),
    latent = Param(
        Fixed(),
        latent_init,
    ),
)
param_HSMM_UV_P5 = (;
    μ = Param(
        [truncated(Normal(-30., 10^5), -100., -20.0), truncated(Normal(0., 10^5), -50., 50.), truncated(Normal(0., 10^5), -50., 50.), truncated(Normal(0., 10^5), -50., 50.), truncated(Normal(50., 10^5), 20.0, 100.0) ],
        [-50., -20., 0.0, 20., 50.],
    ),
    σ = Param(
        [truncated(Normal(5., 10.), 0.0, 10.0), truncated(Normal(2.5, 10.), 0.0, 10.0), truncated(Normal(2.5, 10.), 0.0, 10.0), truncated(Normal(2.5, 10.), 0.0, 10.0), truncated(Normal(2.5, 10.), 0.0, 10.0)],
        [5., 2., 2., 2., 2.],
    ),
    λ = Param(
        [truncated(Normal(5.0, 10^5), 0.0, 10.0), truncated(Normal(10.0, 10^5), 5.0, 20.0),
            truncated(Normal(15.0, 10^5), 0.0, 30.0),truncated(Normal(20.0, 10^5), 10.0, 50.0),
            truncated(Normal(25.0, 10^5), 10.0, 100.0)
        ],
        [5., 10., 15., 20., 25.],
    ),
    p = Param(
        [Dirichlet(4,1/4), Dirichlet(4,1/4), Dirichlet(4,1/4), Dirichlet(4,1/4), Dirichlet(4,1/4)],
        [[.2, .2, .2, .4], [.2, .2, .2, .4], [.2, .2, .2, .4], [.2, .2, .2, .4], [.2, .2, .2, .4]],
    ),
    latent = Param(
        Fixed(),
        latent_init5,
    ),
)

################################################################################
function get_dynamics(model::ModelWrapper{A}, θ) where {A<:Union{HSMM_UV, HSMM_UV3}}
    @unpack μ, σ, r, ϕ, p = θ
    dynamicsᵈ = [ NegativeBinomial(r[iter], ϕ[iter]) for iter in eachindex(μ) ]
    dynamicsᵉ = [ Normal(μ[iter], σ[iter]) for iter in eachindex(μ) ]
    dynamicsˢ = [ Categorical( extend_state(p[iter], iter) ) for iter in eachindex(μ) ]
    return dynamicsᵉ, dynamicsˢ, dynamicsᵈ
end
function get_dynamics(model::ModelWrapper{M}, θ) where {M<:Union{HSMM_UV_P, HSMM_UV_P5}}
    @unpack μ, σ, λ, p = θ
    dynamicsᵈ = [ Poisson(λ[iter]) for iter in eachindex(μ) ]
    dynamicsᵉ = [ Normal(μ[iter], σ[iter]) for iter in eachindex(μ) ]
    dynamicsˢ = [ Categorical( extend_state(p[iter], iter) ) for iter in eachindex(μ) ]
    return dynamicsᵉ, dynamicsˢ, dynamicsᵈ
end

################################################################################
# Define sample and likelihood
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{F}; Nsamples = 1000) where {F<:Union{HSMM_UV, HSMM_UV3, HSMM_UV_P, HSMM_UV_P5}}

    dynamicsᵉ, dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)
    latentⁿᵉʷ = initzeros( convert.(latent_type, (rand(dynamicsˢ[1]), rand(dynamicsᵈ[1]) ) ), Nsamples)
    observedⁿᵉʷ = initzeros( rand(dynamicsᵉ[1]) , Nsamples)

    stateₜ = rand(rng, dynamicsˢ[ rand(1:length(dynamicsˢ) ) ] )
    durationₜ = rand(rng, dynamicsᵈ[stateₜ])

    fillvec!(latentⁿᵉʷ, convert.(latent_type, (stateₜ, durationₜ) ), 1 )
    fillvec!(observedⁿᵉʷ, rand(rng, dynamicsᵉ[stateₜ]), 1 )

    for iter in 2:size(observedⁿᵉʷ,1)
        if durationₜ > 0
            durationₜ -=  1
            fillvec!(latentⁿᵉʷ, convert.(latent_type, (stateₜ, durationₜ) ), iter )
            fillvec!(observedⁿᵉʷ, rand(rng, dynamicsᵉ[stateₜ]), iter )
        else
            stateₜ = rand(rng, dynamicsˢ[ stateₜ ] ) #stateₜ for t-1 overwritten
            durationₜ = rand(rng, dynamicsᵈ[stateₜ])
            fillvec!(latentⁿᵉʷ, convert.(latent_type, (stateₜ, durationₜ) ), iter )
            fillvec!(observedⁿᵉʷ, rand(rng, dynamicsᵉ[stateₜ]), iter )
        end
    end
    return observedⁿᵉʷ, latentⁿᵉʷ
end

function (objective::Objective{<:ModelWrapper{M}})(θ::NamedTuple) where {M<:Union{HSMM_UV, HSMM_UV3, HSMM_UV_P, HSMM_UV_P5}}
    @unpack model, data, tagged = objective
    @unpack latent = θ
## Prior
    lp = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    dynamicsᵉ, dynamicsˢ, dynamicsᵈ = get_dynamics(model, θ)
    ll = 0.0
    sorting = ByRows()
    for iter in 2:size(data,1)
        ll += logpdf(dynamicsᵉ[latent[iter][1]], grab(data, iter, sorting))
        if latent[iter-1][1] != latent[iter][1] #s[iter-1] != s[iter]
            ll += logpdf(dynamicsˢ[latent[iter-1][1]], latent[iter][1])
            ll += logpdf(dynamicsᵈ[latent[iter][1]], latent[iter][2])
        end
    end
    return ll + lp
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Union{HSMM_UV, HSMM_UV3, HSMM_UV_P, HSMM_UV_P5}}
    @unpack model, data = objective
    @unpack μ, σ, p, latent = model.val #s, d
    if latent[end][2] != 0
        s_new = latent[end][1]
    else
        s_new = rand(_rng, Categorical(extend_state(p[latent[end][1]], latent[end][1])))
    end
    return rand(_rng, Normal(μ[s_new], σ[s_new]))
end

################################################################################
#Assign dynamics
function ModelWrappers.dynamics(objective::Objective{<:ModelWrapper{M}}) where {M<:Union{HSMM_UV, HSMM_UV3, HSMM_UV_P, HSMM_UV_P5}}
    @unpack model, data = objective
    dynamicsᵉ, dynamicsˢ, dynamicsᵈ = get_dynamics(model, model.val)

    initialˢ =  Categorical( get_stationary( reduce(hcat, [dynamicsˢ[iter].p for iter in eachindex(dynamicsˢ)] )' ) )
    initialᵈ(sₜ) = dynamicsᵈ[sₜ]
    initial = SemiMarkovInitiation(initialˢ, initialᵈ)

    state(particles, iter) = dynamicsˢ[particles[iter-1][1]]
    duration(s, iter) = dynamicsᵈ[s]
    transition = SemiMarkovTransition(state, duration)

    observation(particles, iter) = dynamicsᵉ[particles[iter][1]]
    return SemiMarkov(initial, transition, observation)
end

################################################################################
hsmm_UV = ModelWrapper(HSMM_UV(), param_HSMM_UV, (;), FlattenDefault(Float64, FlattenContinuous()))
hsmm_UV_P = ModelWrapper(HSMM_UV_P(), param_HSMM_UV_P)
hsmm_UV_P5 = ModelWrapper(HSMM_UV_P5(), param_HSMM_UV_P5)

tagged_hsmm_UV = Tagged(hsmm_UV, (:μ, :σ, :r, :ϕ, :p))
tagged_hsmm_UV_P = Tagged(hsmm_UV_P, (:μ, :σ, :λ, :p))
tagged_hsmm_UV_P5 = Tagged(hsmm_UV_P5, (:μ, :σ, :λ, :p))

data_HSMM_UV, latent_HSMM_UV = simulate(_rng, hsmm_UV; Nsamples = n)
data_HSMM_UV_P, latent_HSMM_UV_P = simulate(_rng, hsmm_UV_P; Nsamples = n)
data_HSMM_UV_P5, latent_HSMM_UV_P5 = simulate(_rng, hsmm_UV_P5; Nsamples = n)

_tagged = Tagged(hsmm_UV, :latent)
fill!(hsmm_UV, _tagged, (; latent = latent_HSMM_UV))
fill!(hsmm_UV_P, _tagged, (; latent = latent_HSMM_UV_P))
fill!(hsmm_UV_P5, _tagged, (; latent = latent_HSMM_UV_P5))

objectiveUV = Objective(hsmm_UV, data_HSMM_UV, tagged_hsmm_UV)
objectiveUV_P = Objective(hsmm_UV_P, data_HSMM_UV_P, tagged_hsmm_UV_P)
objectiveUV_P5 = Objective(hsmm_UV_P5, data_HSMM_UV_P5, tagged_hsmm_UV_P5)

objectiveUV(objectiveUV.model.val)
objectiveUV_P(objectiveUV_P.model.val)
objectiveUV_P5(objectiveUV_P5.model.val)

dynamics(objectiveUV)
dynamics(objectiveUV_P)
dynamics(objectiveUV_P5)

################################################################################
import BaytesInference: filter_forward

"Forward Filter HSMM - target filtering distributions P( sₜ | e₁:ₜ ) instead of forward probabilities P( sₜ, e₁:ₜ ) for numerical stability"
function filter_forward(objective::Objective{<:ModelWrapper{M}}; dmax = size(objective.data, 1)) where {M<:Union{HSMM_UV, HSMM_UV3, HSMM_UV_P, HSMM_UV_P5}}
    @unpack model, data = objective
## Map Parameter to observation and state probabilities
    @unpack p = model.val
    dynamicsᵉ, _, dynamicsᵈ = get_dynamics(model, model.val)
    #!NOTE: Working straight with parameter instead of distributions here as easier to implement
    dynamicsˢ = transpose( reduce(hcat, [  extend_state(p[iter], iter) for iter in eachindex(p) ] ) )
    initialˢ            = get_stationary( dynamicsˢ )
    sorting = ByRows()
## Assign Log likelihood
    ℓℒᵈᵃᵗᵃ = zeros(Float64, size(data,1), size(dynamicsˢ, 1) ) #can be a Matrix instead of Array{k, 3} because eₜ | sₜ, dₜ independent of dₜ
    Base.Threads.@threads for state in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 2) )
    for iter in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 1) )
            ℓℒᵈᵃᵗᵃ[iter, state] = logpdf( dynamicsᵉ[state], grab(data, iter, sorting) )
        end
    end
## Initialize
    Nstates = size(dynamicsˢ, 1)
    α       = zeros( size(ℓℒᵈᵃᵗᵃ, 1), size(ℓℒᵈᵃᵗᵃ, 2), dmax ) # α represents P( sₜ, dₜ | e₁:ₜ ) for each t, where dmax is the maximum duration for a theoretically infinite upper bound for dₜ - numerically more stable than classical forward probabilities p(sₜ, dₜ, e₁ₜ)
    ℓℒ      = 0.0 #Log likelihood container
    c       = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[1,:]) )
## Calculate initial probabilities P( s₁, d₁, e₁ ) ∝ P( s₁ ) * P( d₁ | s₁ ) * P( e₁ | s₁ )
    for stateₜ in Base.OneTo(Nstates) #Start with P( s₁ ) * P( e₁ | s₁ )
        α[1,stateₜ, :] .+= initialˢ[stateₜ] * exp(ℓℒᵈᵃᵗᵃ[1,stateₜ] - c )
        for durationₜ in 0:(dmax-1) #Proceed with P( d₁ | s₁ )
            α[1,stateₜ, durationₜ+1] *= pdf(dynamicsᵈ[ stateₜ ], durationₜ )
        end
    end
## Normalize P( s₁, d₁, e₁ ) and return normalizing constant P( e₁), which can be used to calculate ℓℒ = P(e₁:ₜ) = p(e₁) ∏_k p(eₖ | e₁:ₖ₋₁)
    norm = sum( @view(α[1,:, :]) )
    α[1, :, :] ./= norm
    ℓℒ += log(norm) + c # log(norm) = p(eₜ | e₁:ₜ₋₁), c is constant that is added back after removing on top for numerical stability
## Loop through sequence
    for t = 2:size(α, 1)
        c = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[t,:]) )
        Base.Threads.@threads for stateₜ in Base.OneTo(Nstates)
            for durationₜ in 0:(dmax-1)
                ## First ∑_dₜ₋₁ P(dₜ | sₜ, dₜ₋₁) * (lower term)
                for durationₜ₋₁ in 0:(dmax-1)
                    ## Then calculate ∑_sₜ₋₁ P( sₜ | sₜ₋₁) * P(sₜ₋₁ | e₁:ₜ₋₁) - we sum over sₜ₋₁, which states are per row
                    for stateₜ₋₁ in Base.OneTo(Nstates)
                        if durationₜ₋₁ > 0 #Dirac delta function for duration and state
                            α[t,stateₜ, durationₜ+1] += ( (durationₜ₋₁-1) == durationₜ) * (stateₜ₋₁ == stateₜ) * α[t-1, stateₜ₋₁, durationₜ₋₁+1]
                        else #State and duration probabilities - self-transitions have 0 probabilities if durationₜ₋₁ == 0 via transition[i,i] == 0
                            α[t,stateₜ, durationₜ+1] += pdf(dynamicsᵈ[ stateₜ ], durationₜ) * dynamicsˢ[stateₜ₋₁, stateₜ] * α[t-1, stateₜ₋₁, durationₜ₋₁+1]
                        end
                    end
                end
                ## Then multiply with ℒᵈᵃᵗᵃ corrector P( eₜ | sₜ )
                α[t, stateₜ, durationₜ+1] *= exp(ℓℒᵈᵃᵗᵃ[t,stateₜ] - c ) # - c for higher numerical stability, will be added back to likelihood increment below
            end
        end
        ## Normalize α
        norm = sum( @view(α[t,:, :]) )
        α[t, :, :] ./= norm
        ## Add normalizing constant p(eₜ | e₁:ₜ₋₁) to likelihood term
        ℓℒ += log(norm) + c
    end
    return (α, ℓℒ)
end
