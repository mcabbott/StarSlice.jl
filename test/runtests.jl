using StarSlice, Test

@testset "basics" begin

    A = rand(Int8, 2,3,4)
    @test A[*,:,1] == [A[1,:,1], A[2,:,1]]
    @test A[*,end,:] == [A[1,3,:], A[2,3,:]]

    @test size(first(A[1:2,:,*])) == (2,3)

    B = @view A[1,:,*]
    @test size(B) == (4,)
    @test B[1] isa SubArray
    B[4][2] = 42
    @test A[1,2,4] == 42

    @test first(eachslice(A, :,*,*)) isa Array

    @test_throws BoundsError A[*,:]
    @test_throws BoundsError A[*,:,10]
    @test_throws BoundsError A[1:10,*,:]

end
@testset "dotview" begin

    A = zeros(Int8, 2,3,4)
    B = rand(Int8, 4,3,2)

    @test (A[:,3,*] .= B[*,1,:]) isa Array
    @test A[:,3,:] == transpose(B[:,1,:])

    A[:,*,1] .= [fill(i,2) for i=1:3]
    @test A[:,:,1] == [1,1] .* (1:3)'
    @test all(A[:, 1:2, 2:end] .== 0)

    C = zeros(Int8, 2,4,3);
    @test_throws DimensionMismatch (A[*,:,1] .= C[*,:,1]) # wrong slice size
    @test_throws DimensionMismatch (A[*,:,:] .= C[*,:,:])

    D = zeros(Int8, 2,5,4);
    @test_throws DimensionMismatch (A[1,*,:] .= D[2,*,:]) # wrong outer size

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
