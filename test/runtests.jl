using StarSlice, Test

@testset "basics" begin

    A = rand(Int8, 2,3,4)
    @test A[*,:,1] == [A[1,:,1], A[2,:,1]]
    @test A[*,!,1] == reshape([A[1,:,1], A[2,:,1]], :,1)
    @test A[&,:,1] == [A[1:1,:,1], A[2:2,:,1]]
    @test A[*,end,:] == [A[1,3,:], A[2,3,:]]

    @test size(first(A[1:2,:,*])) == (2,3)
    @test size(first(A[1:2,:,&])) == (2,3,1)

    B = @view A[1,:,*]
    @test size(B) == (4,)
    @test B[1] isa SubArray
    B[4][2] = 42
    @test A[1,2,4] == 42

    @test @views A[*,:,1] == [A[1,:,1], A[2,:,1]]
    @test @views A[*,!,1] == reshape([A[1,:,1], A[2,:,1]], :,1)
    @test @views A[&,:,1] == [A[1:1,:,1], A[2:2,:,1]]
    @test @views A[*,end,:] == [A[1,3,:], A[2,3,:]]

    @test first(eachslice(A, :,*,*)) isa Array

    @test_throws BoundsError A[*,:]
    @test_throws BoundsError A[*,:,10]
    @test_throws BoundsError A[1:10,*,:]
    @test_throws BoundsError @view A[*,:]
    @test_throws BoundsError @view A[*,:,10]
    @test_throws BoundsError @view A[1:10,*,:]

end
@testset "dotview" begin

    A = zeros(Int8, 2,3,4)
    B = rand(Int8, 4,3,2)

    @test (A[:,3,*] .= B[*,1,:]) isa Array
    @test (A[:,3,&] .= B[*,1,:]) isa Array # writes (2,) into (2,1)
    @test A[:,3,:] == transpose(B[:,1,:])

    A[:,*,1] .= [fill(i,2) for i=1:3]
    @test A[:,:,1] == [1,1] .* (1:3)'
    @test all(A[:, 1:2, 2:end] .== 0)

    @test_throws ArgumentError (A[:,3,*] = B[*,1,:]) # = not .=

    @test_throws DimensionMismatch (A[:,3,*] .= B[&,1,:]) # writes (1,2) into (2,)
    C = zeros(Int8, 2,4,3);
    @test_throws DimensionMismatch (A[*,:,1] .= C[*,:,1]) # wrong slice size
    @test_throws DimensionMismatch (A[*,:,:] .= C[*,:,:])
    @test_throws DimensionMismatch (A[*,:,:] .= C[*,:,:])

    D = zeros(Int8, 2,5,4);
    @test_throws DimensionMismatch (A[1,*,:] .= D[2,*,:]) # wrong outer size

    # https://github.com/JuliaLang/julia/issues/16606#issuecomment-522238801
    a, b = rand(3,4), rand(3,4);
    f(x::AbstractVector) = zero(x) .+ length(x)
    @views b[:, *] .= f.(a[:, *])  # foreach((x,y) -> x .= f(y), eachcol(b), eachcol(a))
    @test all(b .== 3)

    @test_broken c[:, *] = f.(a[:, *]) # UndefVarError: c not defined

end
@testset "reduction" begin

    A = rand(1:99, 2,3,4)

    @test sum.(A[&,!,2]) == sum(A[:,:,2], dims=2)
    @test sum.(A[*,:,2]) == vec(sum(A[:,:,2], dims=2))

    @test @views sum.(A[&,!,2]) == sum(A[:,:,2], dims=2)
    @test @views sum.(A[*,:,2]) == vec(sum(A[:,:,2], dims=2))

end
@testset "inference" begin

    A = rand(1:99, 2,3,4)

    @test (@inferred A[*,:,2]; true)
    @test (@inferred view(A,*,:,2); true)
    @test (@inferred first(view(A,*,:,2)); true)
    @test (@inferred eachslice(A,*,:,2); true)
    @test (@inferred first(eachslice(A,*,:,2)); true)

end
@testset "ambiguities" begin

    @test isempty(detect_ambiguities(StarSlice, Base, Core))

end
@info "loading Zygote..."
using Zygote
@testset "gradient" begin

    A = rand(4,3,2)
    G = gradient(A -> sum(prod(A[:,:,1], dims=1)), A)[1]
    @test G ≈ gradient(A -> sum(map(prod, A[:,*,1])), A)[1]

    M = rand(2,3)
    @test gradient(M -> sum(sum(M[*,:])), M) == (ones(2,3),)

    P = rand(2,3,4,5,6);
    G = gradient(P -> sum(prod(P[:,:,1,1,:], dims=1)), P)[1];
    @test G ≈ gradient(P -> sum(map(prod, P[:,*,1,1,*])), P)[1]

end
