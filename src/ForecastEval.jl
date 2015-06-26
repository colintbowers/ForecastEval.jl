module ForecastEval
#-----------------------------------------------------------
#PURPOSE
#	Colin T. Bowers module for forecast evaluation
#NOTES
#LICENSE
#	MIT License (see github repository for more detail: https://github.com/colintbowers/ForecastEval.jl.git)
#-----------------------------------------------------------

#Load any entire modules that are needed (use import ModuleName1, ModuleName2, etc)
using 	StatsBase,
		Distributions,
		LossFunctions,
		DependentBootstrap

#Load any specific variables/functions that are needed (use import ModuleName1.FunctionName1, ModuleName2.FunctionName2, etc)
import 	Base.string,
		Base.show

#Specify the variables/functions to export (use export FunctionName1, FunctionName2, etc)
export 	dieboldmariano,
		DMMethod,
		DMAsymptotic,
		DMBootstrap


#******************************************************************************

#----------------------------------------------------------
#SET CONSTANTS FOR MODULE
#----------------------------------------------------------
#Currently none


#----------------------------------------------------------
#ROUTINE
#	Diebold-Mariano
#OUTPUT
#	The tuple (pval, tailregion), where pval::Float64 is the p-value from the test, and tailregion::Int takes values 0 (fail to reject), 1 (reject in favour of forecast 1), or -1 (reject in favour of forecast 2)
#REFERENCES
#	Diebold, Francis X., Mariano, Roberto S. (1995) "Comparing Predictive Accuracy", Journal of Business and Economic Statistics, 13 (3), pp. 253-263
#NOTES
#	You can skip the block length selection procedure by specifying a block-length > 0
#----------------------------------------------------------
#-------- METHOD TYPES ---------
abstract DMMethod
type DMAsymptotic <: DMMethod
	lagWindow::Int
	function DMAsymptotic(lagWindow::Int)
		lagWindow < 0 && error("Lag window must be non-negative")
		new(lagWindow)
	end
end
DMAsymptotic() = DMAsymptotic(0)
type DMBootstrap <: DMMethod
	bootstrapParam::BootstrapParam
	function DMBootstrap(bootstrapParam::BootstrapParam)
		bootstrapParam.statistic == mean ? new(bootstrapParam) : error("statistic field in BootstrapParam must be set to mean")
	end
end
DMBootstrap(numObs::Int) = DMBootstrap(BootstrapParam(numObs, statistic=mean))
DMBootstrap{T<:Number}(x::Vector{T}) = DMBoostrap(BootstrapParap(x, statistic=mean))
#type methods
Base.string(d::DMAsymptotic) = "dmAsymptotic"
Base.string(d::DMBootstrap) = "dmBootstrap"
function Base.show(io::IO, d::DMAsymptotic)
	println(io, "Diebold-Mariano Asymptotic Method:")
	pritnln(io, "    Lag window = " * string(d.lagWindow))
end
function Base.show(io::IO, d::DMBootstrap)
	println(io, "Diebold-Mariano Bootstrap Method:")
	show(io, d.bootstrapParam)
end
Base.show(d::DMMethod) = show(STDOut, d)
#--------- FUNCTION -----------------------
#Diebold-Mariano test using asymptotic approximation (see section 1.1 of their paper)
function dieboldmariano{T<:Number}(lossDiff::Vector{T}, method::DMAsymptotic; confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	m = mean(lossDiff)
	if method.lagWindow == 0
		v = var(lossDiff)
	else
		v = var(lossDiff) + 2 * sum(autocov(lossDiff, 1:method.lagWindow)) #Rectangular kernel to specified lag
	end
	v <= 0 && (v = 0) #If variance estimate is negative, bind to zero so we auto-reject the null
	testStat = m / sqrt(v / length(lossDiff))
	return(pval(Normal(), testStat, twoSided=true))
end
#Diebold-Mariano test using a dependent bootstrap
function dieboldmariano{T<:Number}(lossDiff::Vector{T}, method::DMBootstrap; confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	getBlockLength(method.bootstrapParam) <= 0 && dbootstrapblocklength!(method.bootstrapParam, lossDiff) #detect block-length if necessary
	method.bootstrapParam.statistic != mean && error("statistic field in BootstrapParam must be set to mean")
	testStat = mean(lossDiff)
	statVec = dbootstrapstatistic(lossDiff, method.bootstrapParam) - testStat #Centre statVec on zero
	sort!(statVec)
	return(pval(statVec, testStat))
end
#Keyword wrapper
function dieboldmariano{T1<:Number, T2<:Number}(lossDiff::Vector{T1}; method::DMMethod=DMBootstrap(length(lossDiff)), numResample::Int=1000, blockLength::T2=-1, lagWindow::Int=0, confLevel::Number=0.05)
	typeof(method) == DMBootstrap && update!(method.bootstrapParam, numResample=numResample, blockLength=blockLength)
	typeof(method) == DMAsymptotic && (method.lagWindow=lagWindow)
	return(dieboldmariano(lossDiff, method, confLevel=confLevel))
end
#Wrapper for calculating loss differential
dieboldmariano{T1<:Number, T2<:Number}(xhat1::Vector{T1}, xhat2::Vector{T1}, x::Vector{T1}, lossFunc::LossFunction=SquaredError(); method::DMMethod=DMBootstrap(length(x)), numResample::Int=1000, blockLength::T2=-1, lagWindow::Int=0, confLevel::Number=0.05) = dieboldmariano(lossdiff(xhat1, xhat2, x, lossFunc), method=method, numResample=numResample, blockLength=blockLength, lagWindow=lagWindow, confLevel=confLevel)





#----------------------------------------------------------
#ROUTINE
#	Reality Check
#OUTPUT
#	Returns a p-value vector, with one p-value for each iteration of the loop over models. Usually it is the last p-value that is of interest.
#REFERENCES
#	White (2000) "A Reality Check for Data Snooping", Econometrica, 68 (5), pp. 1097-1126
#NOTES
#	You can skip the block length selection procedure by specifying a block-length > 0
#----------------------------------------------------------
#-------- METHOD TYPES ---------
abstract RCMethod
type RCBootstrap <: RCMethod
	bootstrapParam::BootstrapParam
	blockLengthFilter::Symbol
	function RCBootstrap(bootstrapParam::BootstrapParam, blockLengthFilter::Symbol)
		!(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) && error("Invalid value for blockLengthFilter")
		new(bootstrapParam, blockLengthFilter)
	end
end
type RCBootstrapAlt <: RCMethod
	bootstrapParam::BootstrapParam
	blockLengthFilter::Symbol
	function RCBootstrapAlt(bootstrapParam::BootstrapParam, blockLengthFilter::Symbol)
		!(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) && error("Invalid value for blockLengthFilter")
		new(bootstrapParam, blockLengthFilter)
	end
end
#type methods
Base.string(x::RCBootstrap) = "rcBootstrap"
function Base.show(io::IO, x::RCBootstrap)
	println(io, "Reality check via bootstrap. Bootstrap parameters are:")
	show(io, x.bootstrapParam)
end
Base.show(x::RCBootstrap) = show(STDOUT, x)
#-------- FUNCTION -------------
#realitycheck for bootstrap method
function realitycheck{T<:Number}(lD::Matrix{T}, method::RCBootstrap)
	numObs = size(lD, 1)
	numModel = size(lD, 2)
	rootNumObs = sqrt(numObs)
	numObs < 3 && error("Not enough observations to perform a reality check")
	numModel < 1 && error("Input data matrix is empty")
	rC_blocklength!(lD, method) #Gets appropriate block-length (if needed)
	method.bootstraParam.numObsData != numObs && error("numObsData field in BootstrapParam is not equal to number of rows in loss differential matrix")
	typeof(method.bootstrapParam.bootstrapMethod) == TaperedBlock && error("realitycheck currently not possible using tapered block bootstrap")
	inds = dbootstrapindex(method.bootstrapParam)
	fBoot = Array(Float64, size(inds, 2))
	vBoot = Array(Float64, size(inds, 2))
	vBoot *= -Inf #All -Inf values will get updated on first iteration
	pValVec = Array(Float64, numModel)
	vBarOld = -Inf #-Inf gets updated on first iteration
	for k = 1:numModel
		fBar = mean(sub(lD, 1:numObs, k))
		vBar = max(vBarOld, rootNumObs * fBar)
		rC_updateboot!(lD, inds, numObs, rootNumObs, k, fBar, fBoot, vBoot)
		println(vBoot) #investigate QuickSort
		sort!(vBoot) #Consider specifying insertion sort (default is QuickSort) since vBar is likely to be close to sorted
		M = searchsortedlast(vBoot, vBar)
		pValVec[k] = 1 - (M / numObs)
		vBarOld = vBar
	end
	return(pValVec)
end
#Non-exported function used to update the \bar{V}_k and \bar{V}_{k,i}^* in White (2000) (see page 1110)
function rC_updateboot!{T<:Number}(lD::Matrix{T}, inds::Matrix{Int}, numObs::Int, rootNumObs::Float64, modelNum::Int, fBar::Float64, fBoot::Vector{Float64}, vBoot::Vector{Float64})
	lDSub = sub(lD, 1:numObs, modelNum)
	for i = 1:size(inds, 2)
		fBoot[i] = mean(lDSub[sub(inds, 1:numObs, i)])
		vBoot[i] = max(vBoot[i], rootNumObs * (fBoot[i] - fBar))
	end
	return(true)
end
#Non-exported function used to choose a block length for bootstrap. Note, if method.bootstrapParam is already > 0, then this routine is skipped.
function rC_blocklength!{T<:Number}(lD::Matrix{T}, method::RCBootstrap)
	if getBlockLength(method.bootstrapParam) <= 0
		if blockLengthFilter == :first
			bL = dbootstrapblocklength(lD[:, 1], method.bootstrapParam)
		else
			blockLengthVec = [ dbootstrapblocklength(lD[:, n], method.bootstrapParam) for n = 1:size(lD, 2) ]
			if blockLengthFilter == :mean;	bL = mean(blockLengthVec)
			elseif blockLengthFilter == :median; bL = median(blockLengthVec)
			elseif blockLengthFilter == :maximum; bL = max(blockLengthVec)
			else; error("Invalid value for blockLengthFilter")
			end
		end
		replaceBlockLength!(method.bootstrapParam, bL)
	else
		bL = convert(Float64, getBlockLength(method.bootstrapParam))
	end
	return(bL)
end
#Alternative bootstrap methodology (duplicated my MatLab code)
function realitycheck{T<:Number}(lD::Matrix{T}, method::RCBootstrapAlt)
	numObs = size(lD, 1)
	numModel = size(lD, 2)
	rootNumObs = sqrt(numObs)
	numResample = method.bootstrapParam.numResample
	numObs < 3 && error("Not enough observations to perform a reality check")
	numModel < 1 && error("Input data matrix is empty")
	rC_blocklength!(lD, method) #Gets appropriate block-length (if needed)
	method.bootstraParam.numObsData != numObs && error("numObsData field in BootstrapParam is not equal to number of rows in loss differential matrix")
	typeof(method.bootstrapParam.bootstrapMethod) == TaperedBlock && error("realitycheck currently not possible using tapered block bootstrap")
	inds = dbootstrapindex(method.bootstrapParam)
	mld = [ mean(lD[:, k]) k = 1:numModel ]
	mldBoot = [ mean(sub(lD, 1:numObs, k)[sub(inds, 1:numObs, i)]) for i = 1:numResample, k = 1:numModel ]
	v = maximum(rootNumObs * mldBoot, 1)
	vBoot = maximum(rootNumObs * (mldBoot .- mld), 2)
	pVal = sum(vBoot .> v) / numResample
	return([pVal])
end
#Keyword wrapper
function realitycheck{T<:Number}(lD::Matrix{T}; method::RCMethod=RCBootstrap(), numResample::Int=1000, blockLength::Number=-1, blockLengthFilter::ASCIIString="mean")
	update!(method.bootstrapParam, numResample=numResample, blockLength=blockLength)
	method.blockLengthFilter = blockLengthFilter
	return(realitycheck(lD, methodIn))
end
#Keyword wrapper (calculates loss differential)
realitycheck{T<:Number}(xPrediction::Matrix{T}, xBaseCase::Vector{T}, x::Vector{T}, lossFunc::LossFunction=SquaredError(); method::RCMethod=RCBootstrap(), numResample::Int=1000, blockLength::Number=-1, blockLengthFilter::ASCIIString="mean") = realitycheck(lossdiff(xPrediction, xBaseCase, xTrue, lossFunc), method=method, numResample=numResample, blockLength=blockLength, blockLengthFilter=blockLengthFilter)






#----------------------------------------------------------
#ROUTINE
#	SPA Test
#OUTPUT
#REFERENCES
#	Hansen (2005) "A Test for Superior Predictive Ability", Journal of Business and Economic Statistics, 23 (4), pp. 365-380
#NOTES
#	You can skip the block length selection procedure by specifying a block-length > 0
#----------------------------------------------------------

















#Function for getting p-values from one-sided or two-sided statistical tests. Input can be eiter sorted vector of iid draws from relevant distribution, or an explicit distribution
function pval{T<:Number}(xVec::Vector{T}, xObs::T; twoSided::Bool=true)
	if twoSided
		i = searchsortedlast(xVec, xObs)
		NHalf = 0.5 * length(xVec)
		if i <= NHalf
			tailRegion = -1
			pv = i / NHalf
		else
			tailRegion = 1
			pv = 2 - (i / NHalf)
		end
	else
		tailRegion = 1
		pv = 1 - (searchsortedlast(xVec, xObs) / length(xVec))
	end
	return(pv, tailRegion)
end
function pval{T<:Number}(d::Distribution, xObs::T; twoSided::Bool=true)
	if twoSided
		if xObs < mean(d)
			tailRegion = -1
			pv = 2*cdf(d, xObs)
		else
			tailRegion = 1
			pv = 2*cdf(d, -1 * xObs)
		end
	else
		tailRegion = 1
		pv = cdf(d, xObs)
	end
	return(pv, tailRegion)
end




end # module




