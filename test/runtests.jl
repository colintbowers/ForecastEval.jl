using Base.Test
using StatsBase, Distributions, DependentBootstrap
using ForecastEval

@testset "Diebold-Mariano tests" begin
	srand(1234)
	x = randn(100)
	y = randn(100)
	z = randn(100)
	ld = (x - z).^2 - (y - z).^2
	srand(1234)
	bO = dm(ld, dmmethod=:boot)
	hO = dm(ld, dmmethod=:hac)
	@test (bO.rejH0, bO.bestinput, bO.tailregion) == (0, 1, 0)
	@test isapprox(bO.pvalue, 0.42599999999999993)
	@test isapprox(bO.teststat, -0.23500198562777386)
	@test (hO.rejH0, hO.bestinput, hO.tailregion) == (0, 1, 0)
	@test isapprox(hO.pvalue, 0.5016201102228858)
	@test isapprox(hO.teststat, -0.6719428022372147)
end

@testset "Reality Check tests" begin
	srand(1234)
	ld = randn(100, 20)
	rcOut = rc(ld)
	@test rcOut.rejH0 == 0
	@test isapprox(rcOut.pvalue, 0.762)
end

@testset "SPA tests" begin
	srand(1234)
	ld = randn(100, 5)
	spaOut = spa(ld)
	@test spaOut.rejH0 == 0
	@test isapprox(sum(spaOut.mu_u), 0.026125507727482858)
	@test isapprox(sum(spaOut.mu_c), 0.026125507727482858)
	@test isapprox(sum(spaOut.mu_l), 0.21071619368258687)
	@test isapprox(spaOut.pvalue_u, 0.25)
	@test isapprox(spaOut.pvalue_c, 0.25)
	@test isapprox(spaOut.pvalue_l, 0.124)
	@test isapprox(spaOut.pvalueauto, 0.25)
	@test isapprox(spaOut.teststat, 1.5384635269840627)
end

@testset "MCS tests" begin
	srand(1234)
	l = randn(100, 8)
	mcsOut = mcs(l)
	@test mcsOut.inQF == Int[5,2,3,7,4,8,6,1]
	@test mcsOut.outQF == Int[]
	@test isapprox(sum(mcsOut.pvalueQF), 6.665)
	@test mcsOut.inMT == Int[5,2,3,7,4,8,6,1]
	@test mcsOut.outMT == Int[]
	@test isapprox(sum(mcsOut.pvalueMT), 6.673)
end
