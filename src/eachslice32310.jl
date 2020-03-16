# https://github.com/JuliaLang/julia/pull/32310/files

struct EachSlice{A,I,L}
    arr::A # underlying array
    cartiter::I # CartesianIndices iterator
    lookup::L # dimension look up: dimension index in cartiter, or nothing
end

function iterate(s::EachSlice, state...)
    r = iterate(s.cartiter, state...)
    r === nothing && return r
    (c,nextstate) = r
    view(s.arr, map(l -> l === nothing ? (:) : c[l], s.lookup)...), nextstate
end

size(s::EachSlice) = size(s.cartiter)
length(s::EachSlice) = length(s.cartiter)
ndims(s::EachSlice) = ndims(s.cartiter)
IteratorSize(::Type{EachSlice{A,I,L}}) where {A,I,L} = IteratorSize(I)
IteratorEltype(::Type{EachSlice{A,I,L}}) where {A,I,L} = EltypeUnknown()

parent(s::EachSlice) = s.arr

function eachrow(A::AbstractVecOrMat)
    iter = CartesianIndices((axes(A,1),))
    lookup = (1,nothing)
    EachSlice(A,iter,lookup)
end
const EachRow{A,I} = EachSlice{A,I,Tuple{Int,Nothing}}

function eachcol(A::AbstractVecOrMat)
    iter = CartesianIndices((axes(A,2),))
    lookup = (nothing,1)
    EachSlice(A,iter,lookup)
end
const EachCol{A,I} = EachSlice{A,I,Tuple{Nothing,Int}}

@inline function eachslice(A::AbstractArray; dims)
    for dim in dims
        dim <= ndims(A) || throw(DimensionMismatch("A doesn't have $dim dimensions"))
    end
    iter = CartesianIndices(map(dim -> axes(A,dim), dims))
    lookup = ntuple(dim -> findfirst(isequal(dim), dims), ndims(A))
    EachSlice(A,iter,lookup)
end
