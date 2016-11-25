
"""
    DMMethod

Abstract type for nesting the various methods that can be used to perform Diebold-Mariano tests. Subtypes are: \n
    DMHAC
    DMBoot \n
The subtypes have entries in the help (?) menu.
"""
abstract DMMethod

"""
    DMHAC(alpha::Float64, kernelfunction::Symbol, bandwidth::Int)
    DMHAC(data ; alpha::Float64=0.05, kernelfunction::Symbol=:epanechnikov, bandwidth::Int=-1)

Method type for doing a Diebold-Mariano test using a HAC variance estimator. Second method is keyword constructor.
The fields of the type follow: \n
    alpha <- Confidence level for the test
    kernelfunction <- Kernel function to use in HAC variance estimator. Valid values are :epanechnikov, :gaussian, :uniform, :bartlett.
    bandwidth <- Bandwidth to use in HAC variance estimator (set less than or equal to -1 to estimate bandwidth using Politis (2003) "Adaptive Bandwidth Choice")
"""
type DMHAC <: DMMethod
    alpha::Float64
    kernelfunction::Symbol
    bandwidth::Int
    function DMHAC(alpha::Float64, kernelfunction::Symbol, bandwidth::Int)
        !(0.0 < alpha < 0.5) && error("Confidence level set to $(alpha) which is not on the (0, 0.5) interval")
        !(any(kernelfunction .== KERNEL_FUNCTIONS)) && error("Invalid kernel function of $(kernelfunction)")
        new(alpha, kernelfunction, bandwidth)
    end
end

"""
    DMBoot(alpha::Float64, bootinput::BootInput)
    DMBoot(data ; alpha::Float64=0.05, blocklength::Number=0.0, numresample::Number=1000, bootmethod::Symbol=:stationary, blmethod::Symbol=:dummy)

Method type for doing a Diebold-Mariano test using a dependent bootstrap. Second method is keyword constructor.
The fields of the type follow: \n
    alpha <- Confidence level for the test
    bootinput <- Specifies type of bootstrap method to use. See DependentBootstrap package or ?dm for more detail.
"""
type DMBoot <: DMMethod
    alpha::Float64
    bootinput::BootInput
    function DMBoot(alpha::Float64, bootinput::BootInput)
        !(0.0 < alpha < 0.5) && error("Confidence level set to $(alpha) which is not on the (0, 0.5) interval")
        new(alpha, bootinput)
    end
end

"""
    DMTest(rejH0::Int, pvalue::Float64, bestinput::Int, teststat::Float64, dmmethod::DMMethod)

Output type for a Diebold-Mariano test. A description of the fields follows: \n
    rejH0 <- true if the null is rejected, false otherwise
    pvalue <- p-value from the test
    bestinput <- 1 if forecast 1 is more accurate, and 2 if forecast 2 is more accurate. See ?dm for definition of forecast 1 versus 2.
    teststat <- If dmmethod == :hac then is the mean of loss difference scaled by HAC variance
                If dmmethod == :boot then is the mean of loss difference
    dmmethod <- Diebold-Mariano method used in the test. See ?DMMethod for more detail.
"""
type DMTest
    rejH0::Bool
    pvalue::Float64
    bestinput::Int
    teststat::Float64
    dmmethod::DMMethod
    DMTest(rejH0::Bool, pvalue::Float64, bestinput::Int, teststat::Float64, dmmethod::DMMethod) = new(rejH0, pvalue, bestinput, teststat, dmmethod)
end

#Constructors for DMHAC
DMHAC( ; alpha::Float64=0.05, kernelfunction::Symbol=:epanechnikov, bandwidth::Int=-1)::DMHAC = DMHAC(alpha, kernelfunction, bandwidth)
DMHAC(data ; alpha::Float64=0.05, kernelfunction::Symbol=:epanechnikov, bandwidth::Int=-1)::DMHAC = DMHAC(alpha, kernelfunction, bandwidth) #Superfluous but included for consistency

#Constructors for DMBoot
DMBoot() = DMBoot(0.01, ())
function DMBoot(data ; alpha::Float64=0.05, blocklength::Number=0.0, numresample::Number=1000, bootmethod::Symbol=:stationary,
                blmethod::Symbol=:dummy)::DMBoot
    return(DMBoot(alpha, BootInput(data, blocklength=blocklength, numresample=numresample, bootmethod=bootmethod, blmethod=blmethod, flevel1=mean)))
end

#Base methods
Base.show(io::IO, d::DMHAC) = println(io, "dmHAC")
Base.show(io::IO, d::DMBoot) = println(io, "dmBoot")

"""
    dm{T<:Number}(lossDiff::Vector{T}, dmm::DMMethod)::DMTest
    dm{T<:Number}(lossDiff::Vector{T} ; alpha::Float64=0.05, dmmethod::Symbol=:boot, blocklength::Number=0.0,
                                        numresample::Number=1000, bootmethod::Symbol=:stationary, blmethod::Symbol=:dummy,
                                        kernelfunction::Symbol=:epanechnikov, bandwidth::Int=-1)

This function implements the test proposed in Diebold, Mariano (1995) "Comparing Predictive Accuracy". \n
Let x_1 denote forecast 1, x_2 denote forecast 2, and let y denote the forecast target. Let L(., .) denote a loss function.
Then the first argument lossDiff is assumed to be a vector created by the following operation: \n
L(x_1, y) - L(x_2, y) \n
It is anticipated that most users will use the keyword method variant of the function signatures. An explanation of the keywords follows: \n
    alpha <- The confidence level of the test
    dmmethod <- The Diebold-Mariano methodology:
            :boot <- Use a block bootstrap specified by blocklength, numresample, bootmethod and blmethod
            :hac <- Use a Gaussian assumption with HAC variance estimator specified by kernelfunction and bandwidth
    blocklength <- If dmmethod = :boot, then gives bootstrap block length. If blocklength <= 0.0 then block length is estimated optimally.
    numresample <- If dmmethod = :boot, then gives number of resamples.
    bootmethod <- If dmmethod = :boot, then gives block bootstrap method.
    blmethod <- If dmmethod = :boot, then gives block length selection method. blmethod = :dummy implies auto-select block length method
    kernelfunction <- if dmmethod = :hac, then gives kernel function to use with HAC variance estimator. See ?hacvariance for more detail.
    bandwidth <- if dmmethod = :hac, then gives bandwidth for HAC variance estimator. If bandwidth <= -1 then bandwidth is estimated using Politis (2003) "Adaptive Bandwidth Choice" \n
For more detail on the bootstrap options and methods, please see the docs for the DependentBootstrap package. \n
The output of a Diebold-Mariano test is of type DMTest. Use ?DMTest for more information.
"""
function dm{T<:Number}(lossDiff::Vector{T}, method::DMHAC)::DMTest
	length(lossDiff) < 2 && error("Input data vector has length of $(length(lossDiff))")
	m = mean(lossDiff)
	(v, _) = hacvariance(lossDiff, kf=method.kernelfunction, bw=method.bandwidth)
	testStat = m / sqrt(v / length(lossDiff))
	pVal = pvaluelocal(Normal(), testStat, tail=:both)
	pVal > method.alpha ? (rejH0 = false) : (rejH0 = true)
    testStat > 0 ? (bestInput = 2) : (bestInput = 1)
	return(DMTest(rejH0, pVal, bestInput, testStat, method))
end
function dm{T<:Number}(lossDiff::Vector{T}, method::DMBoot)::DMTest
    length(lossDiff) < 2 && error("Input data vector has length of $(length(lossDiff))")
    setflevel1!(method.bootinput, mean) #Ensure correct level1 test statistic is used
    statVec = dbootlevel1(lossDiff, method.bootinput)
    pVal = pvaluelocal(statVec, 0.0, tail=:both, as=false)
    pVal > method.alpha ? (rejH0 = false) : (rejH0 = true)
    testStat = mean(lossDiff)
    testStat > 0 ? (bestInput = 2) : (bestInput = 1)
	return(DMTest(rejH0, pVal, bestInput, testStat, method))
end
#Keyword constructor
function dm{T<:Number}(lossDiff::Vector{T} ; alpha::Float64=0.05, dmmethod::Symbol=:boot, blocklength::Number=0.0, numresample::Number=1000,
                       bootmethod::Symbol=:stationary, blmethod::Symbol=:dummy, kernelfunction::Symbol=:epanechnikov, bandwidth::Int=-1)::DMTest
    dmmethod == :hac && return(dm(lossDiff, DMHAC(lossDiff, alpha=alpha, kernelfunction=kernelfunction, bandwidth=bandwidth)))
    dmmethod == :boot && return(dm(lossDiff, DMBoot(lossDiff, alpha=alpha, blocklength=blocklength, numresample=numresample, bootmethod=bootmethod, blmethod=blmethod)))
    error("dmmethod=$(dmmethod) is invalid")
end
#Multivariate method
function dm{T<:Number}(lossDiff::Vector{Vector{T}} ; alpha::Float64=0.05, dmmethod::Symbol=:boot, blocklength::Number=0.0, numresample::Number=1000,
                       bootmethod::Symbol=:stationary, blmethod::Symbol=:dummy, kernelfunction::Symbol=:epanechnikov, bandwidth::Int=-1)::Vector{DMTest}
    dmmethod == :hac && return(DMTest[ dm(lossDiff[k], DMHAC(lossDiff, alpha=alpha, kernelfunction=kernelfunction, bandwidth=bandwidth)) for k = 1:length(lossDiff) ])
    dmmethod == :boot && return(DMTest[ dm(lossDiff[k], DMBoot(lossDiff, alpha=alpha, blocklength=blocklength, numresample=numresample, bootmethod=bootmethod, blmethod=blmethod)) for k = 1:length(lossDiff) ])
    error("dmmethod=$(dmmethod) is invalid")
end
