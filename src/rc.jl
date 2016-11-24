
"""
    RCMethod

Abstract type for nesting the various methods that can be used to perform a Reality Check. Subtypes are: \n
    RCBoot \n
The subtypes have entries in the help (?) menu.
"""
abstract RCMethod

"""
    RCBoot(alpha::Float64, bootinput::BootInput)

Method type for doing a Reality Check using a dependent bootstrap.
The fields of this type follow: \n
    alpha <- Confidence level for the test
    bootinput <- Specifies type of bootstrap method to use. See DependentBootstrap package or ?rc for more detail
"""
type RCBoot <: RCMethod
    alpha::Float64
	bootinput::BootInput
    function RCBoot(alpha::Float64, bootinput::BootInput)
        !(0.0 < alpha < 0.5) && error("Confidence level set to $(alpha) which is not on the (0, 0.5) interval")
        new(alpha, bootinput)
    end
end

"""
    RCTest(rejH0::Int, pvalue::Float64)

Output type from a Reality Check test.
The fields of this type follow: \n
    rejH0 <- true if the null is rejected, false otherwise
    pvalue <- p-value from the test
"""
type RCTest
    rejH0::Bool
    pvalue::Float64
end

#Constructors for RCBoot
function RCBoot(data ; alpha::Float64=0.05, blocklength::Number=0.0, numresample::Number=1000, bootmethod::Symbol=:stationary, blmethod::Symbol=:dummy)::RCBoot
    return(RCBoot(alpha, BootInput(data, blocklength=blocklength, numresample=numresample, bootmethod=bootmethod, blmethod=blmethod)))
end

#Functions for types
Base.show(io::IO, a::RCBoot) = println(io, "rcBoot")

"""
	rc{T<:Number}(lD::Matrix{T}, method::RCBoot)::Float64
    rc{T<:Number}(lD::Matrix{T} ; alpha::Float64=0.05, blocklength::Number=0.0, numresample::Number=1000, bootmethod::Symbol=:stationary, blmethod::Symbol=:dummy)

This function implements the test proposed in White (2000) "A Reality Check for Data Snooping" following the methodology in Hansen (2005). \n
Let x_0 denote a base-case forecast, x_k, k = 1, ..., K, denote K alternative forecasts, and y denote the forecast target.
Let L(., .) denote a loss function. The first argument of rc is a matrix where the kth column of the matrix is created by the operation: \n
L(x_k, y) - L(x_0, y) \n
Note that the forecast loss comes first and the base case loss comes second. This is the opposite to what is described in White's paper.
It is anticipated that most users will use the keyword method variant. An explanation of the keywords follows:
    alpha <- Confidence level of the test
    blocklength <- The bootstrap block length. If blocklength <= 0.0 then block length is estimated.
    numresample <- The bootstrap number of resamples.
    bootmethod <- The block bootstrap method.
    blmethod <- The block length selection method. blmethod = :dummy implies auto-select block length method \n
For more detail on the bootstrap options and methods, please see the docs for the DependentBootstrap package. \n
The output of a Reality Check test is of type RCTest. Use ?RCTest for more information.
"""
function rc{T<:Number}(lD::Matrix{T}, method::RCBoot)::RCTest
    lD *= -1 #White's loss differentials have base case first
	numObs = size(lD, 1)
	numModel = size(lD, 2)
	numResample = method.bootinput.numresample
	numObs < 2 && error("Number of observations = $(numObs) which is not enough to perform a reality check")
	numModel < 1 && error("Input dataset is empty")
	inds = dbootinds(method.bootinput) #Bootstrap indices
    #Get mean loss differentials and bootstrapped mean loss differentials
	mld = Float64[ mean(view(lD, 1:numObs, k)) for k = 1:numModel ]
	mldBoot = Array(Float64, numModel, numResample)
	for j = 1:numResample
		for k = 1:numModel
			mldBoot[k, j] = mean(view(lD, 1:numObs, k)[inds[j]])
		end
	end
	#Get RC test statistic and bootstrapped density under the null
	v = maximum(sqrt(numObs) * mld)
	vBoot = maximum(sqrt(numObs) * (mldBoot .- mld), 1)
	#Calculate p-value and return (as vector)
	pVal = sum(vBoot .> v) / numResample
    pVal < method.alpha ? (rejH0 = true) : (rejH0 = false)
	return(RCTest(rejH0, pVal))
end
#Keyword method
function rc{T<:Number}(lD::Matrix{T} ; alpha::Float64=0.05, blocklength::Number=0.0, numresample::Number=1000, bootmethod::Symbol=:stationary, blmethod::Symbol=:dummy)::RCTest
    return(rc(lD, RCBoot(lD, alpha=alpha, blocklength=blocklength, numresample=numresample, bootmethod=bootmethod, blmethod=blmethod)))
end
