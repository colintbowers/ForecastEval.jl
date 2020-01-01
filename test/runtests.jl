using Test, Random
using StatsBase, Distributions, DependentBootstrap
using ForecastEval

#-------------------------------------------------
#DO NOT CHANGE ANY OF THE FOLLOWING INPUTS
kernelfunctypevec = [ForecastEval.KernelEpanechnikov(), ForecastEval.KernelGaussian(), ForecastEval.KernelBartlett(), ForecastEval.KernelUniform()];
kernelfuncsymvec = [:epanechnikov, :gaussian, :bartlett, :uniform];
bandwidthvec = [-1, 0, 1, 2, 3];
bootmethodsymvec = [:iid, :stationary];
bootmethodtypevec = [DependentBootstrap.BootIID(), DependentBootstrap.BootStationary()]
blocklengthvec = [0.0, 5.0];
xmat =
[-0.371817    0.672154  -1.33346    1.3462       0.321323 ;
 0.0676651  -1.03904    1.12363    0.872374    -0.508987 ;
-1.00231    -0.842998  -1.76564    0.00652721  -1.73871  ;
-0.165286   -0.503325  -1.09233   -0.22975     -0.906645 ;
 0.248638   -0.602089   1.02422    1.24328      1.54605  ;
 0.465047    1.11979    1.09211    0.693409     0.116323 ;
 1.53365     0.968045   0.135202   0.252632    -3.08523  ;
 0.600273   -2.00259   -0.488497   1.17        -1.27601  ;
 1.36719    -0.173315   1.1944    -1.57142      2.36206  ;
 1.22798     0.264002   0.427104  -0.0986267    0.508668 ;
 0.229319    1.6132    -2.14253   -1.58257     -0.530612 ;
 0.0410861  -0.258605  -0.321659  -1.20498     -0.63559  ;
 1.915       1.3493    -0.701988   0.125084     0.683679 ;
 1.11015     1.37006    1.63997   -0.0548267   -0.634976 ;
 0.106914    0.108819   0.535538  -0.073909     1.75006  ;
-0.247781   -1.42433   -0.890601  -0.694821     2.06756  ;
 0.312066    0.885002  -1.39183    0.763129     1.52937  ;
 0.488613   -0.172177  -0.224076  -0.13952     -1.47516  ;
 0.153926    0.422725   0.804834   0.141964     0.761245 ;
-0.679447   -0.502743   1.00025   -0.734587     0.119061];
xmatabs = abs.(xmat);
#-------------------------------------------------



@testset "Diebold-Mariano tests" begin
	#Generate data
	x = xmat[:,1]
	y = xmat[:,2]
	z = xmat[:,3]
	ld = (x - z).^2 - (y - z).^2
	#Test constructor
	for kk = 1:length(kernelfunctypevec)
		for bw in bandwidthvec
			dmhac1 = ForecastEval.DMHAC(alpha=0.05, kernelfunction=kernelfunctypevec[kk], bandwidth=bw)
			@test dmhac1.alpha == 0.05
			@test dmhac1.kernelfunction == kernelfunctypevec[kk]
			@test dmhac1.bandwidth == bw
			dmhac2 = ForecastEval.DMHAC(alpha=0.05, kernelfunction=kernelfuncsymvec[kk], bandwidth=bw)
			@test dmhac1.alpha == 0.05
			@test dmhac1.kernelfunction == kernelfunctypevec[kk]
			@test dmhac1.bandwidth == bw
			dmhac3 = ForecastEval.DMHAC(ld, alpha=0.05, kernelfunction=kernelfunctypevec[kk], bandwidth=bw)
			@test dmhac3.alpha == 0.05
			@test dmhac3.kernelfunction == kernelfunctypevec[kk]
			@test dmhac3.bandwidth == bw
			dmhac4 = ForecastEval.DMHAC(ld, alpha=0.05, kernelfunction=kernelfuncsymvec[kk], bandwidth=bw)
			@test dmhac4.alpha == 0.05
			@test dmhac4.kernelfunction == kernelfunctypevec[kk]
			@test dmhac4.bandwidth == bw
		end
	end
	for kk = 1:length(bootmethodtypevec)
		for bl in blocklengthvec
			dmboot1 = ForecastEval.DMBoot(ld, alpha=0.05, bootmethod=bootmethodtypevec[kk], blocklength=bl)
			@test dmboot1.alpha == 0.05
			@test dmboot1.bootinput.bootmethod == bootmethodtypevec[kk]
			if bl <= 0.0 ; @test dmboot1.bootinput.blocklength > 0.0
			else
				if bootmethodsymvec[kk] == :iid ; @test dmboot1.bootinput.blocklength == 1.0
				else @test dmboot1.bootinput.blocklength == bl
				end
			end
			dmboot2 = ForecastEval.DMBoot(ld, alpha=0.05, bootmethod=bootmethodsymvec[kk], blocklength=bl)
			@test dmboot2.alpha == 0.05
			@test dmboot2.bootinput.bootmethod == bootmethodtypevec[kk]
			if bl <= 0.0 ; @test dmboot2.bootinput.blocklength > 0.0
			else
				if bootmethodsymvec[kk] == :iid ; @test dmboot1.bootinput.blocklength == 1.0
				else @test dmboot1.bootinput.blocklength == bl
				end
			end
		end
	end
	#Test output (hac)
	pvalvec = [0.11440139911128933,0.05985060000192489,0.1211016948931237,0.026592349584142416,0.14442472227131237,0.14442472227131237,0.14442472227131237,0.14442472227131237,0.14442472227131237,0.12870301609991372,0.14442472227131237,0.11825398287003397,0.12491428583032126,0.07056983889420516,0.13149963805876264,0.03576447246936558,0.07469037167648004,0.06832784498031234,0.09957810645179305,0.06041771012189759]
	pvalmat = reshape(pvalvec, (4,5))
	teststatvec = [-1.5787151898146972,-1.8818925954728518,-1.5501659301885555,-2.217450096879554,-1.4595102474128858,-1.4595102474128858,-1.4595102474128858,-1.4595102474128858,-1.4595102474128858,-1.5192363442451382,-1.4595102474128858,-1.562144367590813,-1.534469114610218,-1.808235607809477,-1.5082144973278373,-2.0995951195178453,-1.782361057698249,-1.822838564430011,-1.6469024100931644,-1.8777329254314343]
	teststatmat = reshape(teststatvec, (4,5))
	for kk = 1:length(kernelfuncsymvec)
		for nn = 1:length(bandwidthvec)
			dmout = dm(ld, :hac, alpha=0.05, kernelfunction=kernelfuncsymvec[kk], bandwidth=bandwidthvec[nn])
			@test isapprox(dmout.pvalue, pvalmat[kk,nn])
			@test isapprox(dmout.teststat, teststatmat[kk,nn])
		end
	end
	#Test output (boot)
	Random.seed!(1234)
	pvalmat = [0.094  0.088 ; 0.106  0.026]
	for kk = 1:length(bootmethodsymvec)
		for nn = 1:length(blocklengthvec)
			dmout = dm(ld, :boot, alpha=0.05, bootmethod=bootmethodsymvec[kk], blocklength=blocklengthvec[nn])
			@test isapprox(pvalmat[kk,nn], dmout.pvalue)
		end
	end

end

@testset "Reality Check tests" begin
	#Test constructor
	for kk = 1:length(bootmethodsymvec)
		for nn = 1:length(blocklengthvec)
			rc1 = ForecastEval.RCBoot(xmat, alpha=0.05, bootmethod=bootmethodsymvec[kk], blocklength=blocklengthvec[nn])
			@test rc1.alpha == 0.05
			@test rc1.bootinput.bootmethod == bootmethodtypevec[kk]
			if blocklengthvec[nn] <= 0.0 ; @test rc1.bootinput.blocklength > 0.0
			else
				if bootmethodsymvec[kk] == :iid ; @test rc1.bootinput.blocklength == 1.0
				else @test rc1.bootinput.blocklength == blocklengthvec[nn]
				end
			end
		end
	end
	#Test main routine
	Random.seed!(1234)
	pvalmat = [0.868  0.872 ; 0.879  0.873];
	for kk = 1:length(bootmethodsymvec)
		for nn = 1:length(blocklengthvec)
			rcout = rc(xmat, :boot, alpha=0.05, bootmethod=bootmethodsymvec[kk], blocklength=blocklengthvec[nn])
			@test isapprox(pvalmat[kk,nn], rcout.pvalue)
		end
	end
end

@testset "SPA tests" begin
	#Test constructor
	for kk = 1:length(bootmethodsymvec)
		for nn = 1:length(blocklengthvec)
			for mm = 1:length(kernelfuncsymvec)
				for jj = 1:length(bandwidthvec)
					spa1 = ForecastEval.SPABoot(xmat, alpha=0.05, bootmethod=bootmethodsymvec[kk], blocklength=blocklengthvec[nn], kernelfunction=kernelfuncsymvec[mm], bandwidth=bandwidthvec[jj])
					@test spa1.alpha == 0.05
					@test spa1.bootinput.bootmethod == bootmethodtypevec[kk]
					if blocklengthvec[nn] <= 0.0 ; @test spa1.bootinput.blocklength > 0.0
					else
						if bootmethodsymvec[kk] == :iid ; @test spa1.bootinput.blocklength == 1.0
						else @test spa1.bootinput.blocklength == blocklengthvec[nn]
						end
					end
					@test spa1.kernelfunction == kernelfunctypevec[mm]
					@test spa1.bandwidth == bandwidthvec[jj]
				end
			end
		end
	end
    #Test routine (note, hacvariance has already been fully covered by previous tests, as has bootinput constructor)
    Random.seed!(1234)
    spaout = spa(xmat, alpha=0.05, bootmethod=bootmethodsymvec[2], blocklength=blocklengthvec[1], kernelfunction=kernelfuncsymvec[1], bandwidth=bandwidthvec[1])
    @test isapprox(spaout.pvalue_u, 0.878)
    @test isapprox(spaout.pvalue_c, 0.845)
    @test isapprox(spaout.pvalue_l, 0.783)
    @test isapprox(spaout.pvalueauto, 0.845)
    @test isapprox(spaout.teststat, 0.2827379497687274)
    @test isapprox(sum(spaout.mu_u), -0.42402380049999994)
    @test isapprox(sum(spaout.mu_c), -0.05397999049999995)
    @test isapprox(sum(spaout.mu_l), 0.06876765000000001)
end

@testset "MCS tests" begin
    #Test constructor
    for kk = 1:length(bootmethodsymvec)
        for nn = 1:length(blocklengthvec)
            mca1 = ForecastEval.MCSBoot(xmatabs, alpha=0.05, bootmethod=bootmethodsymvec[kk], blocklength=blocklengthvec[nn], basecaseindex=2)
            @test mca1.alpha == 0.05
            @test mca1.bootinput.bootmethod == bootmethodtypevec[kk]
            if blocklengthvec[nn] <= 0.0 ; @test mca1.bootinput.blocklength > 0.0
            else
                if bootmethodsymvec[kk] == :iid ; @test mca1.bootinput.blocklength == 1.0
                else @test mca1.bootinput.blocklength == blocklengthvec[nn]
                end
            end
            @test mca1.basecaseindex == 2
        end
    end
    for kk = 1:length(bootmethodsymvec)
        for nn = 1:length(blocklengthvec)
            mca1 = ForecastEval.MCSBootLowRAM(xmatabs, alpha=0.05, bootmethod=bootmethodsymvec[kk], blocklength=blocklengthvec[nn], basecaseindex=2)
            @test mca1.alpha == 0.05
            @test mca1.bootinput.bootmethod == bootmethodtypevec[kk]
            if blocklengthvec[nn] <= 0.0 ; @test mca1.bootinput.blocklength > 0.0
            else
                if bootmethodsymvec[kk] == :iid ; @test mca1.bootinput.blocklength == 1.0
                else @test mca1.bootinput.blocklength == blocklengthvec[nn]
                end
            end
            @test mca1.basecaseindex == 2
        end
    end
    #Test routine (note, bootinput constructor has already been fully covered by previous tests)
	Random.seed!(1234)
    mcsout1 = mcs(xmatabs, alpha=0.05, bootmethod=bootmethodsymvec[2], blocklength=blocklengthvec[1], basecaseindex=2)
    @test mcsout1.inQF == Int[3,2,4,1]
    @test mcsout1.outQF == Int[5]
    @test isapprox(sum(mcsout1.pvalueQF), 2.162)
    @test mcsout1.inMT == Int[5,3,2,4,1]
    @test mcsout1.outMT == Int[]
    @test isapprox(sum(mcsout1.pvalueMT), 2.2039999999999997)
    Random.seed!(1234)
    mcsout2 = mcs(xmatabs, ForecastEval.MCSBootLowRAM(xmatabs, alpha=0.05, bootmethod=bootmethodsymvec[2], blocklength=blocklengthvec[1], basecaseindex=2))
    @test mcsout2.inQF == Int[3,2,4,1]
    @test mcsout2.outQF == Int[5]
    @test isapprox(sum(mcsout2.pvalueQF), 2.162)
    @test mcsout2.inMT == Int[5,3,2,4,1]
    @test mcsout2.outMT == Int[]
    @test isapprox(sum(mcsout2.pvalueMT), 2.2039999999999997)
end
