################################################################################
function initzeros(instance::T, Nsamples::Integer) where {T<:Real}
    return zeros(typeof(instance), Nsamples)
end
function initzeros(instance::F, Nsamples::Integer) where {F<:NTuple{k,T} where {k, T} }
    return repeat( [zero.(instance)], Nsamples )
end
function initzeros(instance::Array{F}, Nsamples::Integer) where {F<:Real}
    return zeros(eltype(instance), Nsamples, length(instance) )
end

import Main.BaytesCore: grab

################################################################################
" Fill Matrix or Vector at index index"
function fillcols!(data::Matrix{T}, dataₜ::Vector{T}, index::I) where {T, I<:Integer}
    return data[:, index] = dataₜ
end
function fillrows!(data::Matrix{T}, dataₜ::Vector{T}, index::I) where {T, I<:Integer}
    return data[index,:] = dataₜ
end
function fillvec!(data::Vector{T}, dataₜ::T, index::I) where {T, I<:Integer}
    return data[index] = dataₜ
end
################################################################################
# HMM and HSMM helper for filtering
"Normalize Vector inplace and return normalizing constant"
@inline function normalize!(vec::AbstractVector)
    norm = sum(vec)
    vec ./= norm
    return norm
end
"Faster max function for Viterbi algorithm in HMM. Not exported."
function vec_maximum(vec::AbstractVector)
    m = vec[1]
    @inbounds for i = Base.OneTo(length(vec))
        if vec[i] > m
            m = vec[i]
        end
    end
    return m
end

################################################################################
# HMM and HSMM helper for initial distribution
############################
"Compute Stationary distribution for given transition matrix"
function get_stationary!(Transition::AbstractMatrix{T}) where T<:Real
    # From: https://github.com/QuantEcon/QuantEcon.jl/blob/f454d4dfbaf52f550ddd52eff52471e4b8fddb9d/src/markov/mc_tools.jl
    # Grassmann-Taksar-Heyman (GTH) algorithm (Grassmann, Taksar, and Heyman 1985)
    n = size(Transition, 1)
    x = zeros(T, n)

    @inbounds for k in 1:n-1
        scale = sum(Transition[k, k+1:n])
        if scale <= zero(T)
            # There is one (and only one) recurrent class contained in
            # {1, ..., k};
            # compute the solution associated with that recurrent class.
            n = k
            break
        end
        Transition[k+1:n, k] /= scale

        for j in k+1:n, i in k+1:n
            Transition[i, j] += Transition[i, k] * Transition[k, j]
        end
    end

    # backsubstitution
    x[n] = 1
    @inbounds for k in n-1:-1:1, i in k+1:n
        x[k] += x[i] * Transition[i, k]
    end

    # normalisation
    x /= sum(x)

    return x
end
get_stationary(Transition::AbstractMatrix{T}) where {T<:Real} = get_stationary!(copy(Transition))
