module StarSlice

import Base: getindex, view, dotview, setindex!
if VERSION >= v"1.1"
    import Base: eachslice
else
    import Compat: eachslice
end

import ZygoteRules: _pullback, AContext

using Compat # 2.2 eachslice + 3.1 filter

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

Besides `*`, some other symbols have special meanings:

Dimensions marked with an `&` index the outer array just like `*`,
but are not dropped from the inner array. Thus `size(first(A[&,:])) == (1, size(A,2))`,
each slice is a matrix, instead of `size(first(A[*,:])) == (size(A,2),)`
for which each slice is a vector.

Dimensions marked with an `!` index the inner arrays just like `:`,
but are not dropped from the outer array. Thus `A[*,!]` returns a one-column matrix,
`size(A[*,!]) == (size(A,1), 1)`, instead of a vector `size(A[*,:]) == (size(A,1),)`.

```
B = ones(2,3,4);
B[*,:,*]  # 2×4 Matrix of 3-element Vectors
B[&,!,&]  # 2×1×4 Array of 1×3×1 Arrays
```
"""

@doc star_doc getindex
@doc star_doc view
@doc star_doc eachslice

#===== methods =====#
# One */&/! must appear by the 4th dimension.

_types = [Union{typeof(*), typeof(&), typeof(!)}, Union{Integer, Colon}, AbstractArray]

_count(Ts) = count(isequal(first(_types)), Ts) >= 1

_doubles = Iterators.filter(_count, Iterators.product(_types, _types))
_triples = Iterators.filter(_count, Iterators.product(_types, _types, _types))
_quads   = Iterators.filter(_count, Iterators.product(_types, _types, _types, _types))

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
        f==:getindex && @eval setindex!(A::AbstractArray, val, i::$T, j::$S) = star_set(A, (i,j))
    end
    for (T,S,R) in _triples
        @eval $f(A::AbstractArray, i::$T, j::$S, k::$R) = $g(A, (i,j,k))
        @eval _pullback(::AContext, ::typeof($f), A::AbstractArray, i::$T, j::$S, k::$R) =
            star_adjoint($g, A, (i,j,k))
        f==:getindex && @eval @eval setindex!(A::AbstractArray, val, i::$T, j::$S, k::$R) = star_set(A, (i,j,k))
    end
    for (T,S,R,Q) in _quads
        @eval $f(A::AbstractArray, i::$T, j::$S, k::$R, l::$Q, ms...) = $g(A, (i,j,k,l,ms...))
        @eval _pullback(::AContext, ::typeof($f), A::AbstractArray, i::$T, j::$S, k::$R, l::$Q, ms...) =
            star_adjoint($g, A, (i,j,k,l,ms...))
        f==:getindex && @eval setindex!(A::AbstractArray, val, i::$T, j::$S, k::$R, l::$Q, ms...) = star_set(A, (i,j,k,l,ms...))
    end
    @eval $f(A::AbstractVector, ::typeof(*)) = $f(A, :)
end

function star_set(A::AbstractArray, code::Tuple)
    str = replace(join(string.(code), ", "), "Colon()" => ":")
    throw(ArgumentError("setindex! not defined for special indices * and &, try using broadcasting: A[$str] .= values"))
end

#===== getindex -> collected =====#

@inline function star_arrays(A::AbstractArray, code::Tuple)
    iter = make_iter(A, code)
    [ @inbounds getindex(A, i...) for i in iter ]
end

@inline function make_iter(A::AbstractArray, code::Tuple)
    iters = ntuple(length(code)) do d
        x = code[d]
        (x==*) ? axes(A,d) :
        (x==&) ? (i:i for i in axes(A,d)) :
        (x==!) ? tuple(:) :
        Ref(x)
    end
    @boundscheck begin
        inds = ntuple(length(code)) do d
            x = code[d]
            (x==*)|(x==&)|(x==!) ? (:) : x
        end
        checkbounds(A, inds...)
    end
    Iterators.product(iters...)
end

#===== views =====#

function star_views(A::AbstractArray, code::Tuple)
    iter = make_iter(A, code)
    [ @inbounds view(A, i...) for i in iter ]
end

#===== eachslice -> generator =====#

using UnsafeArrays

function star_iter(A::AbstractArray, code::Tuple)
    iter = make_iter(A, code)
    slice = getindex(A, first(iter)...)
    @uviews A begin
        (copyto!(slice, @inbounds view(A, i...)) for i in iter)
    end
    # ( @inbounds view(A, i...) for i in iter )
end

#===== dotview =====#

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
        x = dv.data[i...]
        dv.data[i...] .= rhs
    end
    dv.data
end

#===== aux =====#

using SparseArrays

# These are included just to resolve ambiguities:
getindex(A::SparseMatrixCSC, ::Colon, ::first(_types)) = star_arrays(A, (:,*))
getindex(A::SparseMatrixCSC, ::first(_types), ::Colon) = star_arrays(A, (*,:))

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
