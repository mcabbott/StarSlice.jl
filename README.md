# StarSlice.jl

This package overloads `getindex` and friends to make slices. 
Inserting `*` means that a given index will label the position in the outer container.
`view` makes views, and `eachslice` makes a generator:

```julia
julia> ones(Int, 2,3,4)[:, *, 4]
3-element Array{Array{Int64,1},1}:
 [1, 1]
 [1, 1]
 [1, 1]

julia> @view ones(Int, 2,3)[*, :]
2-element Array{SubArray{Int64,1,Array{Int64,2},Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true},1}:
 [1, 1, 1]
 [1, 1, 1]

julia> eachslice(ones(Int, 2,3,4,5), *, :, 4, *) |> typeof
Base.Generator{Base.Iterators.ProductIterator{Tuple{Base.OneTo{Int64},Base.RefValue{Colon},
  Base.RefValue{Int64},Base.OneTo{Int64}}},StarSlice.var"#75#76"{Array{Int64,4}}}
```

One use of this is for `mapslices`-like operations. It goes well with [LazyStack.jl](https://github.com/mcabbott/LazyStack.jl):

```julia
julia> using StarSlice, LazyStack

julia> info(x) = [ndims(x), size(x)...]
info (generic function with 1 method)

julia> mapslices(info, rand(10,20,30), dims=(1,3))
3×20×1 Array{Int64,3}:
[:, :, 1] =
  2   2   2   2   2   2   2   2  …   2   2   2   2   2   2   2
 10  10  10  10  10  10  10  10     10  10  10  10  10  10  10
 30  30  30  30  30  30  30  30     30  30  30  30  30  30  30

julia> stack(info, @view rand(10,20,30)[:,*,:])
3×20 Array{Int64,2}:
  2   2   2   2   2   2   2   2  …   2   2   2   2   2   2   2
 10  10  10  10  10  10  10  10     10  10  10  10  10  10  10
 30  30  30  30  30  30  30  30     30  30  30  30  30  30  30
```

Right now the `eachslice` generator returns an `Array` not a view, and re-uses the same one. But this appears to provide few benefits. 

### See also

* [JuliennedArrays.jl](https://github.com/bramtayl/JuliennedArrays.jl)

* [LazyStack.jl](https://github.com/mcabbott/LazyStack.jl)

* [SliceMap.jl](https://github.com/mcabbott/SliceMap.jl), differentiable mapslices.

* [JuliaLang/julia#32310](https://github.com/JuliaLang/julia/pull/32310) about `EachSlice` type

* [tkf/ReduceDims.jl](https://github.com/tkf/ReduceDims.jl) from [JuliaLang/julia#16606](https://github.com/JuliaLang/julia/issues/16606). Compare:

```julia
using ReduceDims, StarSlice
mat = rand(1:99, 3, 4)

s1 = sum(along(mat, &, :)) # lazy mapreduce along "&" directions
sum(mat, dims=1) .== s1    # all equal
sum(mat[*, :]) == sum.(mat[:, *]) == vec(copy(s1))

p1 = prod(along(mat, &, :)) |> copy
prod(mat, dims=1) == p1
prod.(mat[:, *]) == vec(p1) # and prod(mat[*, :]) is an error
```
