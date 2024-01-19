#
# Contains DIIS and ODA interpolation techniques
#
import Base.collect

@doc raw"""
     DIIS(m::Int, iterates::Vector{Any}, residuals::Vector{Any})

Standard fixed depth DIIS as described e.g. in [https://doi.org/10.1051/m2an/2021069]
Note that DIIS is applied in DM conventions.
"""
struct DIIS
    m::Int                            # maximal history size
    iterates::Vector{Any}             # xₙ
    residuals::Vector{Any}            # rₙ
end
DIIS(;m=10) = DIIS(m, [], [])
depth(diis::DIIS) = length(diis.iterates)

# Actualize diis lists and remove old iterates if needed
function Base.push!(diis::DIIS, Pdₙ, Psₙ, Rₙ)
    push!(diis.iterates,  (Pdₙ, Psₙ))
    push!(diis.residuals, vec(Rₙ))
    if depth(diis) > diis.m + 1
        popfirst!(diis.iterates)
        popfirst!(diis.residuals)
    end
    @assert depth(diis) <= diis.m + 1
    @assert length(diis.residuals) == depth(diis)
    diis
end

@doc raw"""
    diis(info)

Returns the DIIS extrapolated densities and corresponding Fock operators
from densities contained in info and history in the DIIS struct.
"""
function (diis::DIIS)(info)
    Pdₙ, Psₙ = info.DMs
    Rₙ = info.∇E
    
    # Special case, no DIIS.
    (diis.m == 0) && return (Pdₙ, Psₙ, Fock_operators(Pdₙ, Psₙ, info.ζ)...)
    # First iteration
    if isempty(diis.iterates)
        push!(diis, Pdₙ, Psₙ, Rₙ)
        return Pdₙ, Psₙ,  Fock_operators(Pdₙ, Psₙ, info.ζ)...
    end

    push!(diis, Pdₙ, Psₙ, Rₙ)
    𝐏 = diis.iterates
    𝐑 = diis.residuals
    𝐒 = hcat([𝐑[i+1] - 𝐑[i] for i in 1:length(𝐑)-1]...)

    # Solve DIIS least square PB
    A = 𝐒'𝐒
    B = 𝐒'𝐑[end]
    C = A\B

    𝐏d_diff = [𝐏[i+1][1] - 𝐏[i][1] for i in 1:length(𝐏)-1]
    𝐏s_diff = [𝐏[i+1][2] - 𝐏[i][2] for i in 1:length(𝐏)-1]
    Pd_diis = Pdₙ - C'𝐏d_diff
    Ps_diis = Psₙ - C'𝐏s_diff
    Pd_diis, Ps_diis, Fock_operators(Pd_diis, Ps_diis, info.ζ)...
end

@doc raw"""
    ODA(densities::Matrix{T}, Fock_operators::Matrix{T}) where {T}

Relaxed constrained interpolation of densities and Fock operators or "ODA".
Have to work with hybrid_scf solver. Kind of bugged for now but used to
work before changing the structure.
"""
mutable struct ODA{T}
    densities::Matrix{T}      # (Pdₙᶜ|Psₙᶜ)
    Fock_operators::Matrix{T} # (Fdₙᶜ|Fsₙᶜ)
end
ODA(;T=Float64) = ODA(T[;;],T[;;])

function collect(oda::ODA)
    DMs = oda.densities
    Focks = oda.Fock_operators
    Nb = size(DMs,1)
    Pd, Ps = DMs[:,1:Nb], DMs[:,Nb+1:end]
    Fd, Fs = Focks[:,1:Nb], Focks[:,Nb+1:end]
    (Pd, Ps), (Fd, Fs)
end

function (oda::ODA{T})(info) where {T<:Real}
    Pdₙ₊₁, Psₙ₊₁ = info.DMs

    # First iteration
    if isempty(oda.densities)
        Fdₙ₊₁, Fsₙ₊₁ = Fock_operators(Pdₙ₊₁, Psₙ₊₁, info.ζ)
        oda.densities = hcat(Pdₙ₊₁, Psₙ₊₁); oda.Fock_operators = hcat(Fdₙ₊₁, Fsₙ₊₁)
        return (Pdₙ₊₁, Psₙ₊₁, Fdₙ₊₁, Fsₙ₊₁)
    end

    # Subsequent iterations
    (Pdₙᶜ, Psₙᶜ), (Fdₙᶜ, Fsₙᶜ) = collect(oda)

    # Compute new convex combination
    c₁, c₂, Fdₙ₊₁, Fsₙ₊₁ = oda_polynom_coefficients(Pdₙᶜ, Psₙᶜ, Fdₙᶜ, Fsₙᶜ,
                                                    Pdₙ₊₁, Psₙ₊₁, info.ζ)
    t = oda_convex_combination_param(c₁, c₂) # Add changing guess if t=0
    Pdₙ₊₁ᶜ = (1-t)*Pdₙᶜ + t*Pdₙ₊₁
    Psₙ₊₁ᶜ = (1-t)*Psₙᶜ + t*Psₙ₊₁
    Fdₙ₊₁ᶜ = (1-t)*Fdₙᶜ + t*Fdₙ₊₁
    Fsₙ₊₁ᶜ = (1-t)*Fsₙᶜ + t*Fsₙ₊₁
    oda.densities = hcat(Pdₙ₊₁ᶜ , Psₙ₊₁ᶜ); oda.Fock_operators = hcat(Fdₙ₊₁ᶜ ,Fsₙ₊₁ᶜ)
    Pdₙ₊₁ᶜ, Psₙ₊₁ᶜ, Fdₙ₊₁ᶜ, Fsₙ₊₁ᶜ
end

"""
Find the t_min ∈ [0,1] that minimizes the polynom c₁*x^2 + c₂*x + c₃
Note that t_min is independant of c₃
"""
function oda_convex_combination_param(c₁::T, c₂::T) where {T<:Real}
    t_min = zero(Int64)
    t_extremum = -c₂/(2*c₁) # t that gives the global min or max of the polynom
    if c₁ > 0
        (t_extremum ≤ 0) && (t_min = zero(T))
        (t_extremum ≥ 1) && (t_min = one(T))
        (0 < t_extremum < 1) && (t_min = t_extremum)
    elseif c₁ == 0
        (c₂ > 0) && (t_min = zero(T))
        (c₂ < 0) && (t_min = one(T))
    else #c₁ < 0
        (t_extremum < 0.5) && (t_min = one(T))
        (t_extremum > 0.5) && (t_min = zero(T))
        (t_extremum == 0.5) && (t_min = t_extremum)
    end
    iszero(t_min) && (@warn "Coefficient of the convex combination is 0.."*
                      " Changin guess of the subproblem.")
    @show t_min
end
"""
   oda_polynom_coefficients(Pdₙᶜ::Matrix{T}, Psₙᶜ::Matrix{T}, Fdₙᶜ::Matrix{T},
                                  Fsₙᶜ::Matrix{T}, Pdₙ₊₁::Matrix{T}, Psₙ₊₁::Matrix{T},
                                  ζ::State) where {T<:Real}

Build c₁, c₂ such that
```math
  E((1-t)(Pdₙᶜ,Psₙᶜ) + t(Pdₙ₊₁,Psₙ₊₁)) = c₁*t² + c₂*t + c₃.
```
We have:
  - ``c₁ = (Pdₙ₊₁-Pdₙᶜ, Psₙ₊₁-Psₙᶜ) ⋅ (Fdₙ₊₁-Fdₙᶜ, Fsₙ₊₁-Fsₙᶜ)``
  - ``c₂ = 2(Pdₙ₊₁-Pdₙᶜ, Psₙ₊₁-Psₙᶜ) ⋅ (Fd, Fs)``
  - ``c₃ = E(Pdₙᶜ, Psₙᶜ)`` is not usefull hence not computed

Also returns Fdₙ₊₁ and Fsₙ₊₁ to avoid unnecessary computations.
"""
function oda_polynom_coefficients(Pdₙᶜ::Matrix{T}, Psₙᶜ::Matrix{T}, Fdₙᶜ::Matrix{T},
                                  Fsₙᶜ::Matrix{T}, Pdₙ₊₁::Matrix{T}, Psₙ₊₁::Matrix{T},
                                  ζ::State) where {T<:Real}
    Fdₙ₊₁, Fsₙ₊₁ = Fock_operators(Pdₙ₊₁, Psₙ₊₁, ζ)
    Md =  Pdₙ₊₁ - Pdₙᶜ; Ms = Psₙ₊₁ - Psₙᶜ
    c₁ = tr(Md'*(Fdₙ₊₁-Fdₙᶜ) + Ms'*(Fsₙ₊₁-Fsₙᶜ))
    c₂ = 2*tr(Md'Fdₙᶜ + Ms'Fsₙᶜ)
    c₁, c₂, Fdₙ₊₁, Fsₙ₊₁
end
