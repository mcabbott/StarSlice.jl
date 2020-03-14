
using StarSlice, LazyStack, BenchmarkTools

f(x) = (100*ndims(x) + length(x))  .+ x .^ 2
Z = reshape(1:30, 3,5,2) .+ 0;

mapslices(f, Z, dims=(1,2))
reshape(reduce(hcat, map(f, eachslice(Z, dims=3))), size(Z))
stack(f, Z[:,:,*])

@code_warntype Z[:,:,*]
@code_warntype map(f, Z[:,:,*])

@code_warntype eachslice(Z, dims=3) # Body::Base.Generator{Base.OneTo{Int64},_A} where _A
@code_warntype map(f, eachslice(Z, dims=3))

# This in fact re-uses the same slice!
dodgy = stack(map(identity, eachslice(Z, :,*,1)))
dodgy.slices[1][1] = 99
dodgy

#====== REPL =====#
# An entirely trivial example:

julia> A = rand(10,10,10);

julia> @btime mapslices(identity, $A, dims=(1,2));
  10.679 μs (97 allocations: 12.27 KiB)

julia> @btime reduce(hcat, map(identity, eachslice($A, dims=3)));
  3.061 μs (16 allocations: 8.80 KiB)

julia> @btime stack(map(identity, $A[:,:,*])); # make slices + lazy container
  1.314 μs (21 allocations: 9.22 KiB)

julia> @btime stack(identity, $A[:,:,*]); # write slices into a new Array
  2.277 μs (27 allocations: 17.33 KiB)

julia> @btime stack(map(identity, @view $A[:,:,*])); # make views + lazy container
  243.470 ns (21 allocations: 1.09 KiB)

julia> @btime stack(identity, @view $A[:,:,*]); # write views into new Array
  1.160 μs (27 allocations: 9.20 KiB)

julia> @btime stack(map(identity, eachslice($A, :,:,*)));
  198.967 ns (20 allocations: 960 bytes)

julia> @btime stack(identity, eachslice($A, :,:,*));
  1.071 μs (26 allocations: 9.05 KiB)


# Was hoping that this might avoid generic matmul etc...

@which *(M, M)
@which *(M, view(B, :,1,:))

julia> const M = randn(20,20);

julia> g(x) = sum(transpose(x) * M * x)
g (generic function with 1 method)

julia> B = randn(20,20,20);

julia> @btime mapslices(g, $B, dims=(1,3));
  81.850 μs (338 allocations: 141.80 KiB)

julia> @btime map(g, eachslice($B, dims=2));
  48.871 μs (65 allocations: 131.56 KiB)

julia> @btime map(g, $B[:,*,:]);
  55.508 μs (70 allocations: 195.61 KiB)

julia> @btime map(g, @view $B[:,*,:]);
  48.114 μs (70 allocations: 131.86 KiB)

julia> @btime map(g, eachslice($B, :,*,:));
  56.557 μs (51 allocations: 133.67 KiB)

# again with slice on first dim?
# Here the views are terrible, presumably because of generic matmul

julia> @btime mapslices(g, $B, dims=(2,3));
  91.953 μs (338 allocations: 141.80 KiB)

julia> @btime map(g, eachslice($B, dims=1));
  234.152 μs (145 allocations: 135.94 KiB)

julia> @btime map(g, $B[*,:,:]);
  56.904 μs (69 allocations: 195.59 KiB)

julia> @btime map(g, @view $B[*,:,:]);
  233.534 μs (149 allocations: 136.22 KiB)

julia> @btime map(g, eachslice($B, *,:,:));
  55.576 μs (49 allocations: 133.64 KiB)


julia> const M2 = randn(50,50); # bigger
julia> g2(x) = sum(transpose(x) * M2 * x)
julia> B2 = randn(50,50,50);

julia> @btime map(g2, $B2[*,:,:]);
  1.219 ms (309 allocations: 2.88 MiB)

julia> @btime map(g2, @view $B2[*,:,:]);
  10.175 ms (559 allocations: 1.94 MiB)

julia> @btime map(g2, eachslice($B2, *,:,:));
  1.215 ms (210 allocations: 1.94 MiB)

julia> const M4 = zeros(50,50); # with cache
julia> const M5 = zeros(50,50);
julia> g3(x) = sum(mul!(M5, x', mul!(M4, M2, x)));

julia> @btime map(g3, $B2[*,:,:]);
  1.111 ms (109 allocations: 983.13 KiB)

julia> @btime map(g3, @view $B2[*,:,:]);
  10.084 ms (359 allocations: 20.63 KiB)

julia> @btime map(g3, eachslice($B2, *,:,:));
  1.111 ms (10 allocations: 20.28 KiB)

# Even here, the speed gain from re-using slice is minimal.

# ================


parent_type(A) = typeof(parent(A))
# parent_type(::Type{T}) where {T} = first(Base.return_types(parent, Tuple{T})) # works but slow

    # res = first(Base.return_types(parent, Tuple{T})) # ERROR: code reflection cannot be used from generated functions

    # obj = T(undef, ntuple(_ -> 1, ndims(T))) # only works when undef is defined
    # res = typeof(parent(obj))

@generated parent_type(::Type{T}) where {T} =
    haskey(parent_dict, T) ? :( $(parent_dict[T]) ) : :( push_parent_type(T) )
push_parent_type(T) =
    get!(parent_dict, T) do
        res = first(Base.return_types(parent, Tuple{T}))
    end
const parent_dict = Dict{Type, Type}()
@btime parent_type($(transpose(rand(3)) |> typeof))
@btime parent_type($(view(rand(3,4),:,1) |> typeof)) # 157.593 ns (1 allocation: 16 bytes)
# not free, as dict wasn't full when function got generated
