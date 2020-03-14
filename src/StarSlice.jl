module StarSlice

import Base: getindex, view, dotview
if VERSION >= v"1.1"
    import Base: eachslice
else
    import Compat: eachslice
end

import ZygoteRules: _pullback, AContext

star_doc = """
    A[*,:]
    @view A[:,*]
    eachslice(A, :,*)

Slicing methods added by StarSlice.jl, returning an array indexed by the dimensions marked `*`,
whose elements each fix the value of these, and slice the rest as shown.

```
A[*,:] == [A[i,:] for i in axes(A,1)] == collect.(eachcol(A))

B[1:2, :, *] == [B[1:2, :, k] for k in axes(B,3)]

@view A[:,*] == [view(A,:,j) for j in axes(A,2)] == collect(eachrow(A))

eachslice(A, :,*) ≈ eachcol(A) # Generator
```
"""

@doc star_doc getindex
@doc star_doc view
@doc star_doc eachslice

_types = [typeof(*), Union{Integer, Colon}, AbstractArray]

_doubles = [(T,S) for T in _types for S in _types if T==typeof(*) || S==typeof(*)]

_triples = [(T,S,R) for T in _types for S in _types for R in _types]
_triples = filter(t -> count(isequal(typeof(*)), t) > 0, _triples)

_quads = [(T,S,R,Q) for T in _types for S in _types for R in _types for Q in _types]
_quads = filter(t -> count(isequal(typeof(*)), t) > 0, _quads)

for (f,g) in [
    (:getindex, :star_arrays),
    (:view, :star_views),
    (:dotview, :star_dotviews),
    (:eachslice, :star_iter),
    ]
    for (T,S) in _doubles
        @eval $f(A::AbstractArray, i::$T, j::$S) = $g(A, (i,j))
        @eval _pullback(::AContext, ::typeof($f), A::AbstractArray, i::$T, j::$S) =
            star_adjoint($g, A, (i,j))
    end
    for (T,S,R) in _triples
        @eval $f(A::AbstractArray, i::$T, j::$S, k::$R) = $g(A, (i,j,k))
        @eval _pullback(::AContext, ::typeof($f), A::AbstractArray, i::$T, j::$S, k::$R) =
            star_adjoint($g, A, (i,j,k))
    end
    for (T,S,R,Q) in _quads
        @eval $f(A::AbstractArray, i::$T, j::$S, k::$R, l::$Q, ms...) = $g(A, (i,j,k,l,ms...))
        @eval _pullback(::AContext, ::typeof($f), A::AbstractArray, i::$T, j::$S, k::$R, l::$Q, ms...) =
            star_adjoint($g, A, (i,j,k,l,ms...))
    end
    @eval $f(A::AbstractVector, ::typeof(*)) = $f(A, :)
end

function star_arrays(A::AbstractArray, code::Tuple)
    iter = make_iter(A, code)
    [ @inbounds getindex(A, i...) for i in iter ]
end

function star_views(A::AbstractArray, code::Tuple)
    iter = make_iter(A, code)
    [ @inbounds view(A, i...) for i in iter ]
end

using UnsafeArrays

function star_iter(A::AbstractArray, code::Tuple)
    iter = make_iter(A, code)
    slice = getindex(A, first(iter)...)
    @uviews A begin
        (copyto!(slice, @inbounds view(A, i...)) for i in iter)
    end
    # ( @inbounds view(A, i...) for i in iter )
end

@inline function make_iter(A::AbstractArray, code::Tuple)
    iters = ntuple(length(code)) do d
        x = code[d]
        (x==*) ? axes(A,d) : Ref(x)
    end
    @boundscheck begin
        inds = ntuple(length(code)) do d
            x = code[d]
            (x==*) ? first(axes(A,d)) : x
        end
        checkbounds(A, inds...)
    end
    Iterators.product(iters...)
end

_string(code::Tuple) = replace(string(code), "Colon()" => ":")

struct StarDotview{T,S}
    data::T
    iter::S
end

function star_dotviews(A::AbstractArray, code::Tuple)
    iter = make_iter(A, code)
    StarDotview(A, iter)
end

Base.size(dv::StarDotview) = size(dv.iter)

function Base.copyto!(dv::StarDotview, bc::Base.Broadcast.Broadcasted)
    size(dv) == size(bc) || throw(DimensionMismatch("outer dimensions of target array must match right hand side"))
    foreach(dv.iter, bc) do i, rhs
        dv.data[i...] .= rhs
    end
    dv.data
end

using SparseArrays

# These are included just to resolve ambiguities:
getindex(A::SparseMatrixCSC, ::Colon, ::typeof(*)) = star_arrays(A, (:,*))
getindex(A::SparseMatrixCSC, ::typeof(*), ::Colon) = star_arrays(A, (*,:))

#=
using ZygoteRules

# This never gets called, has to be caught earlier?
@adjoint function star_arrays(A::AbstractArray, code::Tuple)
    println("ok")
    star_arrays(A, code), Δ -> (zero(A)[code...] .= Δ, nothing)
end
=#

function star_adjoint(f::Function, A::AbstractArray, code::Tuple)
    f(A, code), Δ -> (nothing, zero(A)[code...] .= Δ, map(_->nothing, code)...)
end

end # module
