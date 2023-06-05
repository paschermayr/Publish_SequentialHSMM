################################################################################
struct HMM_UV <: ModelName end

################################################################################
#UNIVARIATE CASE
latent_init = convert.(latent_type, rand( Categorical(3), n) )

param_HMM_UV = (;
    μ = Param(
        [truncated(Normal(-.1, 10^5), -10., 0.0), truncated(Normal(.1, 10^5), 0.0, 10.0) ],
        [-.1, .1],
    ),
    σ = Param(
        [truncated(Normal(1.5, 10^5), 0.0, 10.0), truncated(Normal(.5, 10^5), 0.0, 10.0)],
        [1.3, 0.5],
    ),
    p = Param(
        [Dirichlet(2,2) for i in 1:2],
        [[.7, .3], [.05, .95]],
    ),
	latent = Param(
        [Categorical(2) for _ in 1:n],
        rand( Categorical(2), n),
    ),
)

################################################################################
function get_dynamics(model::ModelWrapper{<:HMM_UV}, θ)
    @unpack μ, σ, p = θ
    dynamicsᵉ = [ Normal(μ[iter], σ[iter]) for iter in eachindex(μ) ]
    dynamicsˢ = [ Categorical( p[iter] ) for iter in eachindex(μ) ]
    return dynamicsᵉ, dynamicsˢ
end

################################################################################
# Define sample and likelihood
function ModelWrappers.simulate(rng::Random.AbstractRNG, model::ModelWrapper{F}; Nsamples = 1000) where {F<:Union{HMM_UV}}
    dynamicsᵉ, dynamicsˢ = get_dynamics(model, model.val)
    latentⁿᵉʷ = initzeros( convert(latent_type, rand(dynamicsˢ[1]) ), Nsamples)
    observedⁿᵉʷ = initzeros( rand(dynamicsᵉ[1]) , Nsamples)

    stateₜ = convert(latent_type, rand( dynamicsˢ[ rand(1:length(dynamicsˢ) ) ] ) )
    fillvec!(latentⁿᵉʷ, stateₜ, 1 )
    fillvec!(observedⁿᵉʷ, rand(dynamicsᵉ[stateₜ]), 1 )

    for iter in 2:size(observedⁿᵉʷ,1)
            stateₜ = convert(latent_type, rand( dynamicsˢ[ stateₜ ] ) ) #stateₜ for t-1 overwritten
            fillvec!(latentⁿᵉʷ, stateₜ, iter )
            fillvec!(observedⁿᵉʷ, rand(dynamicsᵉ[stateₜ]), iter )
    end
    return observedⁿᵉʷ, latentⁿᵉʷ
end

function (objective::Objective{<:ModelWrapper{M}})(θ::NamedTuple) where {M<:Union{HMM_UV}}
    @unpack model, data, tagged = objective
    @unpack latent = θ
## Prior
    lp = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    dynamicsᵉ, dynamicsˢ = get_dynamics(model, θ)
    ll = 0.0
    structure = ByRows()
    for iter in 2:size(data,1)
        ll += logpdf( dynamicsᵉ[ latent[iter] ], grab(data, iter, structure) )
        ll += logpdf( dynamicsˢ[latent[iter-1]], latent[iter] )
    end
    return ll + lp
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Union{HMM_UV}}
    @unpack model, data = objective
    @unpack μ, σ, p, latent = model.val
    latent_new = rand(_rng, Categorical(p[latent[end]]))
    return rand(_rng, Normal(μ[latent_new], σ[latent_new]))
end

################################################################################
#Assign dynamics
#Univariate
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{M}}) where {M<:Union{HMM_UV}}
    @unpack model, data = objective
    dynamicsᵉ, dynamicsˢ = get_dynamics(model, model.val)

    initialˢ =  Categorical( get_stationary( reduce(hcat, [dynamicsˢ[iter].p for iter in eachindex(dynamicsˢ)] )' ) )
    transition(particles, iter) = dynamicsˢ[particles[iter-1]] #sₜ::R, eₜ::S, iter::Int64) where {R, S}                = dynamicsˢ[sₜ]
    evidence(particles, iter) = dynamicsᵉ[particles[iter]] #observation(sₜ::R, eₜ₋₁::S, iter::Int64) where {R, S}   = dynamicsᵉ[sₜ]#[zₜ[1]]

    return Markov(initialˢ, transition, evidence)
end

################################################################################
hmm_UV = ModelWrapper(HMM_UV(), param_HMM_UV)

tagged_hmm_UV = Tagged(hmm_UV, (:μ, :σ, :p) )

data_HMM_UV, latent_HMM_UV = simulate(_rng, hmm_UV; Nsamples = n)
_tagged = Tagged(hmm_UV, :latent)
fill!(hmm_UV, _tagged, (; latent = latent_HMM_UV))

objectiveUV = Objective(hmm_UV, data_HMM_UV,  Tagged(hmm_UV, (:μ, :σ, :p)))
objectiveUV(hmm_UV.val)
predict(_rng, objectiveUV)
dynamics(objectiveUV)

################################################################################
import BaytesInference: filter_forward

"Forward Filter HMM"
function filter_forward(objective::Objective{<:ModelWrapper{M}}) where {M<:Union{HMM_UV}}
    @unpack model, data = objective
## Map Parameter to observation and state probabilities
    @unpack p = model.val
    dynamicsᵉ, _ = get_dynamics(model, model.val)
#!NOTE: Working straight with parameter instead of distributions here as easier to implement
    dynamicsˢ           = transpose( reduce(hcat, p ) ) #[ Categorical(p[iter]) for iter in eachindex(μ) ]
    initialˢ            = get_stationary( dynamicsˢ )
    structure = ByRows()
## Assign Log likelihood
    ℓℒᵈᵃᵗᵃ = zeros(Float64, size(data,1), size(dynamicsˢ, 1) )
    Base.Threads.@threads for state in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 2) )
    for iter in Base.OneTo( size(ℓℒᵈᵃᵗᵃ, 1) )
            ℓℒᵈᵃᵗᵃ[iter, state] += logpdf( dynamicsᵉ[state], grab(data, iter, structure) )
        end
    end
## Initialize
    Nstates = size(dynamicsˢ, 1)
    α = zeros( size(ℓℒᵈᵃᵗᵃ) ) # α represents P( sₜ | e₁:ₜ ) for each t, which is numerically more stable than classical forward probabilities p(sₜ, e₁:ₜ)
    c = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[1,:]) ) # Used for stable logsum() calculations for incremental log likelihood addition, i.e.: log( exp(x) + exp(y) ) == x + log(1 + exp(y -x ) ) for y >= x.
    ℓℒ = 0.0 #Log likelihood container
## Calculate initial probabilities
    for stateₜ in Base.OneTo(Nstates)
         α[1,stateₜ] += initialˢ[stateₜ] * exp(ℓℒᵈᵃᵗᵃ[1,stateₜ]-c) #Calculate initial p(s₁, e₁) ∝ P(s₁) * p(e₁ | s₁)
    end
## Normalize P( s₁, e₁ ) and return normalizing constant P(e₁), which can be used to calculate ℓℒ = P(e₁:ₜ) = p(e₁) ∏_k p(eₖ | e₁:ₖ₋₁)
    norm = sum( @view(α[1,:]) )
    α[1,:] ./= norm
    ℓℒ += log(norm) + c# log(norm) = p(eₜ | e₁:ₜ₋₁), c is constant that is added back after removing on top for numerical stability
## Loop through sequence
    for t = 2:size(α, 1)
        c = vec_maximum( @view(ℓℒᵈᵃᵗᵃ[t,:]) ) # Used for stable logsum() calculations, i.e.: log( exp(x) + exp(y) ) == x + log(1 + exp(y -x ) ) for y >= x.
        ## Calculate ∑_sₜ₋₁ P( sₜ | sₜ₋₁) * P(sₜ₋₁ | e₁:ₜ₋₁) - we sum over sₜ₋₁ - states are per row
        for stateₜ in Base.OneTo(Nstates)
            for stateₜ₋₁ in Base.OneTo(Nstates)
                α[t,stateₜ] += dynamicsˢ[stateₜ₋₁, stateₜ] * α[t-1,stateₜ₋₁] # * for both log and non-log version as inside log( sum(probability(...)))
            end
            ## Then multiply with ℒᵈᵃᵗᵃ corrector P( eₜ | sₜ )
            α[t,stateₜ] *= exp(ℓℒᵈᵃᵗᵃ[t,stateₜ]-c) # - c for higher numerical stability, will be added back to likelihood increment below
        end
        ## Normalize α and obtain P( eₜ | e₁:ₜ₋₁)
        norm = sum( @view(α[t,:]) )
        α[t,:] ./= norm
        ## Add normalizing constant p(eₜ | e₁:ₜ₋₁) to likelihood term
        ℓℒ += log(norm)+c
    end
    return (α, ℓℒ)
end
