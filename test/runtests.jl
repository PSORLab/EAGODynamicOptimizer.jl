using Test, EAGODynamicOptimizer

@testset "State Vector" begin
    traj = EAGODynamicOptimizer.Trajectory{Float64}(5, 2)
    @test length(traj.v[1]) == 2
    @test length(traj.v) == 5
end
