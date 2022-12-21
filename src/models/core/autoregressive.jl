################################################################################
# ARHMM and ARHSMM helper for getting Mu

#UNIVARIATE CASE - Data either scalar or Vector, weights either scalar or vector - only 2 cases
#1 Scalar to Scalar
"Return additional parameter weights based on past data"
function get_ARweights(distr::UnivariateDistribution, dataₜ₋ₖ::T, weights::F) where {F<:Real, T<:Real}
#!NOTE: Case where UV Data has Memory 1
    return dataₜ₋ₖ * weights
end
#2 Vector to Vector ~ @view SubArray of Vector is Vector, so Vector{T} is sufficient
function get_ARweights(distr::UnivariateDistribution, dataₜ₋ₖ::AbstractVector{T}, weights::Vector{F}) where {F<:Real, T<:Real}
#!NOTE: Case where UV Data has Memory >1
    return sum( dataₜ₋ₖ .* weights )
end

#MULTIVARIATE CASE - DATA always Matrix, but simplifies to vector if AR1 memory if data[t-1,:] -> so need to dispatch via distributional Form, weights only implemented for AR1 case
function get_ARweights(distr::MultivariateDistribution, dataₜ₋ₖ::AbstractVector{T}, weights::Vector{F}) where {F<:Real, T<:Real}
#!NOTE: Case where MV Data has Memory 1
    return vec(  dataₜ₋ₖ .* weights   )
end
function get_ARweights(distr::MultivariateDistribution, dataₜ₋ₖ::AbstractVector{T}, weights::F) where {F<:Real, T<:Real}
#!NOTE: Case where MV Data has Memory 1 and equal memory
    return dataₜ₋ₖ * weights
end

################################################################################
"Return data subsequence based on memory"
function grab_sequence(data::AbstractArray{R}, iter::I, memory::I) where {R<:Real, I<:Integer}
    if memory > 1
        return grab(data, (iter-memory):(iter-1) )
    else
        return grab(data, (iter-1) )
    end
end
################################################################################
"Return time dependent mean parameter"
function get_μₜ(distr::UnivariateDistribution, dataₜ₋ₖ::D, weights::W) where {D,W}
    return distr.μ + get_ARweights(distr, dataₜ₋ₖ, weights)
end
function get_μₜ(distr::MultivariateDistribution, dataₜ₋ₖ::D, weights::W) where {D,W}
    return distr.μ .+ get_ARweights(distr, dataₜ₋ₖ, weights)
end
