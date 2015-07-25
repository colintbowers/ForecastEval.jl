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
		KernelStat,
		DependentBootstrap

#Load any specific variables/functions that are needed (use import ModuleName1.FunctionName1, ModuleName2.FunctionName2, etc)
import 	Base.string,
		Base.show,
		DependentBootstrap.update!

#Specify the variables/functions to export (use export FunctionName1, FunctionName2, etc)
export 	ForecastEvalMethod,
		ForecastEvalMethodNoBaseCase,
		ForecastEvalMethodWithBaseCase,
		dm,
		DMMethod,
		DMAsymptoticBasic,
		DMAsymptoticHAC,
		DMBootstrap,
		rc,
		RCMethod,
		RCBootstrap,
		spa,
		SPAMethod,
		SPABootstrap



#******************************************************************************

#----------------------------------------------------------
#SET CONSTANTS FOR MODULE
#----------------------------------------------------------
#Currently none

#Set overall abstract type
abstract ForecastEvalMethod
abstract ForecastEvalMethodWithBaseCase <: ForecastEvalMethod
abstract ForecastEvalMethodNoBaseCase <: ForecastEvalMethod



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
abstract DMMethod <: ForecastEvalMethodNoBaseCase
#Asymptotic approximation with basic HAC estimator
type DMAsymptoticBasic <: DMMethod
	lagWindow::Int
	function DMAsymptoticBasic(lagWindow::Int)
		lagWindow < 0 && error("Lag window must be non-negative")
		new(lagWindow)
	end
end
DMAsymptoticBasic() = DMAsymptoticBasic(0)
#Asymptotic approximation with custom HAC estimator
type DMAsymptoticHAC <: DMMethod
	hacMethod::HACVarianceMethod
	DMAsymptoticHAC(hacMethod::HACVarianceMethod) = new(hacMethod)
end
DMAsymptoticHAC() = DMAsymptoticHAC(HACVarianceBasic())
DMAsymptoticHAC(numObs::Int) = DMAsymptoticHAC(HACVarianceBasic(numObs))
DMAsymptoticHAC{T<:Number}(lD::Vector{T}) = DMAsymptoticHAC(HACVarianceBasic(length(lD)))
#Bootstrap approximation
type DMBootstrap <: DMMethod
	bootstrapParam::BootstrapParam
	function DMBootstrap(bootstrapParam::BootstrapParam)
		bootstrapParam.statistic == mean ? new(bootstrapParam) : error("statistic field in BootstrapParam must be set to mean")
	end
end
DMBootstrap(numObs::Int) = DMBootstrap(BootstrapParam(numObs, statistic=mean))
DMBootstrap{T<:Number}(x::Vector{T}) = DMBoostrap(BootstrapParap(x, statistic=mean))
#type methods
Base.string(d::DMAsymptoticBasic) = "dmAsymptoticBasic"
Base.string(d::DMAsymptoticHAC) = "dmAsymptoticHAC"
Base.string(d::DMBootstrap) = "dmBootstrap"
function deepcopy(x::DMMethod)
	tempArgs = [ deepcopy(getfield(x, i)) for i = 1:length(names(x)) ]
	return(eval(parse(string(typeof(x)) * "(tempArgs...)")))
end
function Base.show(io::IO, d::DMBootstrap)
	println(io, "Diebold-Mariano method = " * string(d))
	show(d.bootstrapParam)
end

function Base.show(io::IO, d::DMMethod)
	println(io, "Diebold-Mariano method = " * string(d))
	fieldNames = names(d)
	for n = 1:length(fieldNames)
		println("    field " * string(fieldNames[n]) * " = " * string(getfield(d, n)))
	end
end
Base.show(d::DMMethod) = show(STDOUT, d)
#update! methods
function update!(method::DMAsymptoticBasic; lagWindow::Int=-999)
	lagWindow != -999 && (method.lagWindow = lagWindow)
	return(method)
end
#--------- FUNCTION -----------------------
#Diebold-Mariano test using asymptotic approximation and basic HAC estimator (see section 1.1 of their paper)
function dm{T<:Number}(lossDiff::Vector{T}, method::DMAsymptoticBasic; confLevel::Number=0.05)
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
	(pVal, tailRegion) = pvalue(Normal(), testStat, twoSided=true)
	pVal > confLevel && (tailRegion = 0)
	return(pVal, tailRegion)
end
#Diebold-Mariano test using asymptotic approximation and custom HAC estimator (see section 1.1 of their paper)
function dm{T<:Number}(lossDiff::Vector{T}, method::DMAsymptoticHAC; confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	m = mean(lossDiff)
	(v, _) = hacvariance(lossDiff, method.hacMethod)
	v <= 0 && (v = 0) #If variance estimate is negative, bind to zero so we auto-reject the null
	testStat = m / sqrt(v / length(lossDiff))
	(pVal, tailRegion) = pvalue(Normal(), testStat, twoSided=true)
	pVal > confLevel && (tailRegion = 0)
	return(pVal, tailRegion)
end
#Diebold-Mariano test using a dependent bootstrap
function dm{T<:Number}(lossDiff::Vector{T}, method::DMBootstrap; confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	getblocklength(method.bootstrapParam) <= 0 && dbootstrapblocklength!(method.bootstrapParam, lossDiff) #detect block-length if necessary
	update!(method.bootstrapParam, numObsData=length(lossDiff), numObsResample=length(lossDiff), statistic=mean) #Ensure correct fixed values in BootstrapParam
	testStat = mean(lossDiff)
	statVec = dbootstrapstatistic(lossDiff, method.bootstrapParam) - testStat #Centre statVec on zero
	sort!(statVec)
	(pVal, tailRegion) = pvalue(statVec, testStat)
	pVal > confLevel && (tailRegion = 0)
	return(pVal, tailRegion)
end
#Keyword wrapper
function dm{T1<:Number}(lossDiff::Vector{T1}; method::DMMethod=DMBootstrap(length(lossDiff)), numResample::Int=-999, blockLength::Number=-999, lagWindow::Int=-999, confLevel::Number=0.05)
	typeof(method) == DMBootstrap && update!(method.bootstrapParam, numResample=numResample, blockLength=blockLength)
	typeof(method) == DMAsymptoticBasic && update!(method, lagWindow=lagWindow)
	return(dm(lossDiff, method, confLevel=confLevel))
end
#Wrapper for calculating loss differential
dm{T1<:Number}(xhat1::Vector{T1}, xhat2::Vector{T1}, x::Vector{T1}; lossFunction::LossFunction=SquaredLoss(), method::DMMethod=DMBootstrap(length(x)), numResample::Int=-999, blockLength::Number=-999, lagWindow::Int=-999, confLevel::Number=0.05) = dm(lossdiff(xhat1, xhat2, x, lossFunction), method=method, numResample=numResample, blockLength=blockLength, lagWindow=lagWindow, confLevel=confLevel)
#Method for multiple forecasts against one base-case
function dm{T1<:Number}(lossDiff::Matrix{T1}, method::DMMethod ; confLevel::Number=0.05)
	size(lossDiff, 2) < 1 && error("Not enough input series to perform test")
	pValVec = Array(Float64, size(lossDiff, 2))
	tailRegionVec = Array(Int, size(lossDiff, 2))
	for k = 1:size(lossDiff, 2)
		(pValVec[k], tailRegionVec[k]) = dm(lossDiff[:, k], method, confLevel=confLevel)
	end
	return(pValVec, tailRegionVec, mean(lossDiff, 1))
end
#Keyword wrappers for multiple forecasts
function dm{T1<:Number}(lossDiff::Matrix{T1}; method::DMMethod=DMBootstrap(length(lossDiff)), numResample::Int=-999, blockLength::Number=-999, lagWindow::Int=-999, confLevel::Number=0.05)
	typeof(method) == DMBootstrap && update!(method.bootstrapParam, numResample=numResample, blockLength=blockLength)
	typeof(method) == DMAsymptoticBasic && update!(method, lagWindow=lagWindow)
	return(dm(lossDiff, method, confLevel=confLevel))
end
#Wrapper for calculating loss differential when there are multiple forecasts
dm{T1<:Number}(xPrediction::Matrix{T1}, xBaseCase::Vector{T1}, xTrue::Vector{T1}; lossFunction::LossFunction=SquaredLoss(), method::DMMethod=DMBootstrap(length(x)), numResample::Int=-999, blockLength::Number=-999, lagWindow::Int=-999, confLevel::Number=0.05) = dm(lossdiff(xPrediction, xBaseCase, xTrue, lossFunction), method=method, numResample=numResample, blockLength=blockLength, lagWindow=lagWindow, confLevel=confLevel)






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
abstract RCMethod <: ForecastEvalMethodWithBaseCase
type RCBootstrap <: RCMethod
	bootstrapParam::BootstrapParam
	blockLengthFilter::Symbol
	function RCBootstrapAlt(bootstrapParam::BootstrapParam, blockLengthFilter::Symbol)
		!(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) && error("Invalid value for blockLengthFilter")
		new(bootstrapParam, blockLengthFilter)
	end
end
RCBootstrap(numObs::Int) = RCBootstrapAlt(BootstrapParam(numObs), :median)
#type methods
Base.string(x::RCBootstrap) = "rcBootstrap"
function deepcopy(x::RCMethod)
	tempArgs = [ deepcopy(getfield(x, i)) for i = 1:length(names(x)) ]
	return(eval(parse(string(typeof(x)) * "(tempArgs...)")))
end
function Base.show(io::IO, x::RCBootstrap)
	println(io, "Reality check via bootstrap. Bootstrap parameters are:")
	show(io, x.bootstrapParam)
	println("    Block length filter = " * string(x.blockLengthFilter))
end
Base.show(x::RCBootstrap) = show(STDOUT, x)
function update!(x::RCBootstrap; blockLengthFilter::Symbol=:none)
	if blockLengthFilter != :none
		(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) ? (x.blockLengthFilter = blockLengthFilter) : error("Invalid symbol for blockLengthFilter field")
	end
	return(x)
end
#-------- FUNCTION -------------
#rc for bootstrap method (note, this follows Hansen (2005) method, so only one p-value is output at the end)
function rc{T<:Number}(lD::Matrix{T}, method::RCBootstrap)
	#White's loss differentials have base case first
	lD *= -1
	#Validate inputs
	numObs = size(lD, 1)
	rootNumObs = sqrt(numObs)
	numModel = size(lD, 2)
	numResample = method.bootstrapParam.numResample
	numObs < 3 && error("Not enough observations to perform a reality check")
	numModel < 1 && error("Input data matrix is empty")
	method.bootstrapParam.numObsData != numObs && error("numObsData field in BootstrapParam is not equal to number of rows in loss differential matrix")
	typeof(method.bootstrapParam.bootstrapMethod) == BootstrapTaperedBlock && error("reality check currently not possible using tapered block bootstrap")
	#Get block length and bootstrap indices
	getblocklength(method.bootstrapParam) <= 0 && multivariate_blocklength!(lD, method.bootstrapParam, method.blockLengthFilter)
	inds = dbootstrapindex(method.bootstrapParam)
	#Get mean loss differentials and bootstrapped mean loss differentials
	mld = Float64[ mean(sub(lD, 1:numObs, k)) for k = 1:numModel ]
	mldBoot = Array(Float64, numModel, numResample)
	for j = 1:numResample
		for k = 1:numModel
			mldBoot[k, j] = mean(sub(lD, 1:numObs, k)[sub(inds, 1:numObs, j)])
		end
	end
	#Get RC test statistic and bootstrapped density under the null
	v = maximum(rootNumObs * mld)
	vBoot = maximum(rootNumObs * (mldBoot .- mld), 1)
	#Calculate p-value and return (as vector)
	pVal = sum(vBoot .> v) / numResample
	return(pVal)
end
#Keyword wrapper
function rc{T<:Number}(lD::Matrix{T}; method::RCMethod=RCBootstrap(size(lD, 1)), numResample::Int=-999, blockLength::Number=-999, blockLengthFilter::Symbol=:none)
	update!(method.bootstrapParam, numResample=numResample, blockLength=blockLength)
	update!(method, blockLengthFilter=blockLengthFilter)
	return(rc(lD, method))
end
#Keyword wrapper (calculates loss differential)
rc{T<:Number}(xPrediction::Matrix{T}, xBaseCase::Vector{T}, xTrue::Vector{T}; lossFunction::LossFunction=SquaredLoss(), method::RCMethod=RCBootstrap(length(x)), numResample::Int=-999, blockLength::Number=-999, blockLengthFilter::Symbol=:none) = rc(lossdiff(xPrediction, xBaseCase, xTrue, lossFunction), method=method, numResample=numResample, blockLength=blockLength, blockLengthFilter=blockLengthFilter)






#----------------------------------------------------------
#ROUTINE
#	SPA Test
#OUTPUT
#REFERENCES
#	Hansen (2005) "A Test for Superior Predictive Ability", Journal of Business and Economic Statistics, 23 (4), pp. 365-380
#NOTES
#	You can skip the block length selection procedure by specifying a block-length > 0
#----------------------------------------------------------
#-------- METHOD TYPES ---------
abstract SPAMethod <: ForecastEvalMethodWithBaseCase
type SPABootstrap <: SPAMethod
	bootstrapParam::BootstrapParam
	blockLengthFilter::Symbol
	hacVarianceMethod::HACVarianceMethod
	muMethod::Symbol
	function SPABootstrap(bootstrapParam::BootstrapParam, blockLengthFilter::Symbol, hacVarianceMethod::HACVarianceMethod, muMethod::Symbol)
		!(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) && error("Invalid value for blockLengthFilter")
		!(muMethod == :lower || muMethod == :centre || muMethod == :upper || muMethod == :all || muMethod == :auto) && error("Invalid symbol for muMethod field")
		new(bootstrapParam, blockLengthFilter, hacVarianceMethod, muMethod)
	end
end
function SPABootstrap(numObs::Int; hacVariant::Symbol=:none, muMethod::Symbol=:none, blockLength::Number=-1)
	x = SPABootstrap(BootstrapParam(numObs, blockLength=blockLength), :mean, HACVarianceBasic(KernelPR1994SB(numObs, 0.1), BandwidthMax()), :auto)
	update!(x, numObs, hacVariant=hacVariant, muMethod=muMethod)
	return(x)
end
#type methods
function deepcopy(x::SPAMethod)
	tempArgs = [ deepcopy(getfield(x, i)) for i = 1:length(names(x)) ]
	return(eval(parse(string(typeof(x)) * "(tempArgs...)")))
end
Base.string(x::SPABootstrap) = "spaBootstrap"
function Base.show(io::IO, x::SPABootstrap)
	println(io, "Test for Superior Predictive Ability via bootstrap. Bootstrap parameters are:")
	show(io, x.bootstrapParam)
	println(io, "Block length filter is:" * string(x.blockLengthFilter))
	println(io, "HAC variance method is:")
	show(io, x.hacVarianceMethod)
	println(io, "mu method is:" * string(x.muMethod))
end
Base.show(x::SPABootstrap) = show(STDOUT, x)
function update!(x::SPABootstrap, numObs::Int; hacVariant::Symbol=:none, muMethod::Symbol=:none, blockLengthFilter::Symbol=:none)
	if hacVariant == :basic; x.hacVarianceMethod = HACVarianceBasic(KernelPR1994SB(numObs, 0.1), BandwidthMax())
	elseif hacVariant == :epanechnikov; x.hacVarianceMethod = HACVarianceBasic(KernelEpanechnikov(1.0), BandwidthP2003(numObs))
	elseif hacVariant == :bartlett; x.hacVarianceMethod = HACVarianceBasic(KernelBartlett(1.0), BandwidthP2003(numObs))
	elseif hacVariant != :none; error("hacVariant symbol not recognised")
	end
	if muMethod != :none
		(muMethod == :lower || muMethod == :centre || muMethod == :upper || muMethod == :all || muMethod == :auto) ? (x.muMethod = muMethod) : error("Invalid symbol for muMethod field")
	end
	if blockLengthFilter != :none
		(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) ? (x.blockLengthFilter = blockLengthFilter) : error("Invalid symbol for blockLengthFilter field")
	end
	return(x)
end
#-------- FUNCTION -------------
#spa for bootstrap method
function spa{T<:Number}(lD::Matrix{T}, method::SPAMethod)
	#Hansen's loss differentials have base case first
	lD *= -1
	#Validate inputs
	numObs = size(lD, 1)
	rootNumObs = sqrt(numObs)
	numModel = size(lD, 2)
	numResample = method.bootstrapParam.numResample
	numObs < 3 && error("Not enough observations to perform an SPA test")
	numModel < 1 && error("Input data matrix is empty")
	method.bootstrapParam.numObsData != numObs && error("numObsData field in BootstrapParam is not equal to number of rows in loss differential matrix")
	typeof(method.bootstrapParam.bootstrapMethod) == BootstrapTaperedBlock && error("spa test currently not possible using tapered block bootstrap")
	#Get block length and bootstrap indices
	getblocklength(method.bootstrapParam) <= 0 && multivariate_blocklength!(lD, method.bootstrapParam, method.blockLengthFilter)
	inds = dbootstrapindex(method.bootstrapParam)
	#Get hac variance estimators
	if typeof(method.hacVarianceMethod) == HACVarianceBasic
		if typeof(method.hacVarianceMethod.kernelFunction) == KernelPR1994SB
			pNew = 1 / getblocklength(method.bootstrapParam) #If using Politis, Romano (1994) "The Stationary Bootstrap" HAC estimator, set geometric distribution parameter using the chosen block length
			if pNew <= 0.0; pNew = 0.01
			elseif pNew >= 1.0; pNew = 0.99
			end
			method.hacVarianceMethod.kernelFunction.p = pNew
		end
	end
	wSqVec = Array(Float64, numModel)
	for k = 1:numModel
		(wSqVec[k], _) = hacvariance(lD[:, k], method.hacVarianceMethod)
		wSqVec[k] <= 0.0 && (wSqVec[k] = nextfloat(0.0))
	end
	wInv = Float64[ 1 / sqrt(wSqVec[k]) for k = 1:numModel ]
	#Get bootstrapped mean loss differentials
	mldBoot = Array(Float64, numModel, numResample)
	for j = 1:numResample
		for k = 1:numModel
			mldBoot[k, j] = mean(sub(lD, 1:numObs, k)[sub(inds, 1:numObs, j)])
		end
	end
	#Get mu definitions
	mu_u = Float64[ mean(sub(lD, 1:numObs, k)) for k = 1:numModel ]
	mu_l = Float64[ max(mu_u[k], 0) for k = 1:numModel ]
	multTerm_mu_c = (1/numObs) * 2 * log(log(numObs))
	mu_c = Float64[ (mu_u[k] >= -1 * sqrt(multTerm_mu_c * wSqVec[k])) * mu_u[k] for k = 1:numModel ]
	#Build vector of the mu definitions we plan to use
	if method.muMethod == :lower; muVecVec = Array(Vector{Float64}, 1); muVecVec[1] = mu_l
	elseif method.muMethod == :centre; muVecVec = Array(Vector{Float64}, 1); muVecVec[1] = mu_c
	elseif method.muMethod == :upper; muVecVec = Array(Vector{Float64}, 1); muVecVec[1] = mu_u
	elseif method.muMethod == :all; muVecVec = Array(Vector{Float64}, 3); muVecVec[1] = mu_u; muVecVec[2] = mu_c; muVecVec[3] = mu_l
	elseif method.muMethod == :auto #Automatically detect which p-value to use
		muVecVec = Array(Vector{Float64}, 1)
		any(mu_c .!= 0.0) ? muVecVec[1] = mu_c : muVecVec[1] = mu_u
	else; error("Invalid muMethod field")
	end
	#Get test statistic
	tSPA = maximum([ max(rootNumObs * mu_u[k] * wInv[k], 0) for k = 1:numModel ])
	#Loop over muVecVec and get the p-value for each muMethod
	pValVec = Array(Float64, length(muVecVec))
	for q = 1:length(muVecVec)
		z = rootNumObs * (wInv .* (mldBoot .- muVecVec[q]))
		tSPAmu = Float64[ max(0, maximum(sub(z, 1:numModel, i))) for i = 1:numResample ]
		pValVec[q] = (1 / numResample) * sum(tSPAmu .> tSPA)
	end
	return(pValVec)
end
#Keyword wrapper
function spa{T<:Number}(lD::Matrix{T}; method::SPAMethod=SPABootstrap(size(lD, 1)), numResample::Int=-999, blockLength::Number=-999, blockLengthFilter::Symbol=:none, hacVariant::Symbol=:none, muMethod::Symbol=:none)
	update!(method.bootstrapParam, numResample=numResample, blockLength=blockLength)
	update!(method, size(lD, 1), hacVariant=hacVariant, muMethod=muMethod, blockLengthFilter=blockLengthFilter)
	return(spa(lD, method))
end
#Keyword wrapper (calculates loss differential)
spa{T<:Number}(xPrediction::Matrix{T}, xBaseCase::Vector{T}, xTrue::Vector{T}; lossFunction::LossFunction=SquaredLoss(), method::SPAMethod=SPABootstrap(size(lD, 1)), numResample::Int=-999, blockLength::Number=-999, blockLengthFilter::Symbol=:none, hacVariant::Symbol=:none, muMethod::Symbol=:none) = spa(lossdiff(xPrediction, xBaseCase, xTrue, lossFunction), method=method, numResample=numResample, blockLength=blockLength, blockLengthFilter=blockLengthFilter, hacVariant=hacVariant, muMethod=muMethod)









#non-exported method for obtaining the appropriate block-length for a multivariate time-series
function multivariate_blocklength!{T<:Number}(x::Matrix{T}, bp::BootstrapParam, blockLengthFilter::Symbol)
	if blockLengthFilter == :first
		bL = dbootstrapblocklength(x[:, 1], bp.bootstrapParam)
		blockLengthVec = [bL]
	else
		blockLengthVec = [ dbootstrapblocklength(x[:, n], bp) for n = 1:size(x, 2) ]
		if blockLengthFilter == :mean;	bL = mean(blockLengthVec)
		elseif blockLengthFilter == :median; bL = median(blockLengthVec)
		elseif blockLengthFilter == :maximum; bL = maximum(blockLengthVec)
		elseif blockLengthFilter == :minimum; bL = minimum(blockLengthVec)
		else; error("Invalid value for blockLengthFilter")
		end
	end
	update!(bp.bootstrapMethod, blockLength=bL)
	return(bL, blockLengthVec)
end








#Function for getting p-values from one-sided or two-sided statistical tests. Input can be eiter sorted vector of iid draws from relevant distribution, or an explicit distribution
function pvalue{T<:Number}(xVec::Vector{T}, xObs::T; twoSided::Bool=true)
	if twoSided
		i = searchsortedlast(xVec, xObs)
		NHalf = 0.5 * length(xVec)
		if i <= NHalf
			tailRegion = 1
			pv = i / NHalf
		else
			tailRegion = -1
			pv = 2 - (i / NHalf)
		end
	else
		tailRegion = 1
		pv = 1 - (searchsortedlast(xVec, xObs) / length(xVec))
	end
	return(pv, tailRegion)
end
function pvalue{T<:Number}(d::Distribution, xObs::T; twoSided::Bool=true)
	if twoSided
		if xObs < mean(d)
			tailRegion = 1
			pv = 2*cdf(d, xObs)
		else
			tailRegion = -1
			pv = 2*cdf(d, -1 * xObs)
		end
	else
		tailRegion = 1
		pv = cdf(d, xObs)
	end
	return(pv, tailRegion)
end




end # module




#-------- IMPLEMENTATION OF REALITY CHECK FOLLOWING METHOD IN SECTION 3 OF WHITE'S PAPER (CURRENTLY BROKEN)
# Base.string(x::RCBootstrapLong) = "rcBootstrapLong"
# type RCBootstrapLong <: RCMethod
# 	bootstrapParam::BootstrapParam
# 	blockLengthFilter::Symbol
# 	function RCBootstrapLong(bootstrapParam::BootstrapParam, blockLengthFilter::Symbol)
# 		!(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) && error("Invalid value for blockLengthFilter")
# 		new(bootstrapParam, blockLengthFilter)
# 	end
# end
# RCBootstrapLong(numObs::Int) = RCBootstrapLong(BootstrapParam(numObs), :median)
# function rc{T<:Number}(lD::Matrix{T}, method::RCBootstrap)
# 	#White's loss differentials have base case first
# 	lD *= -1
# 	#Validate inputs
# 	numObs = size(lD, 1)
# 	numModel = size(lD, 2)
# 	rootNumObs = sqrt(numObs)
# 	numResample = method.bootstrapParam.numResample
# 	numObs < 3 && error("Not enough observations to perform a reality check")
# 	numModel < 1 && error("Input data matrix is empty")
# 	#Get block-length and bootstrap indices
# 	getblocklength(method.bootstrapParam) <= 0 && multivariate_blocklength!(lD, method.bootstrapParam, method.blockLengthFilter)
# 	method.bootstrapParam.numObsData != numObs && error("numObsData field in BootstrapParam is not equal to number of rows in loss differential matrix")
# 	typeof(method.bootstrapParam.bootstrapMethod) == BootstrapTaperedBlock && error("rc currently not possible using tapered block bootstrap")
# 	inds = dbootstrapindex(method.bootstrapParam)
# 	#Perform reality check
# 	fBoot = Array(Float64, size(inds, 2))
# 	vBoot = Array(Float64, size(inds, 2))
# 	vBoot *= -Inf #All -Inf values will get updated on first iteration
# 	pValVec = Array(Float64, numModel)
# 	vBarOld = -Inf #-Inf gets updated on first iteration
# 	for k = 1:numModel
# 		fBar = mean(sub(lD, 1:numObs, k))
# 		vBar = max(vBarOld, rootNumObs * fBar)
# 		rc_updateboot!(lD, inds, rootNumObs, k, fBar, fBoot, vBoot)
# 		#println(vBoot) #investigate QuickSort
# 		sort!(vBoot) #Consider specifying insertion sort (default is QuickSort) since vBar is likely to be close to sorted
# 		M = searchsortedlast(vBoot, vBar)
# 		pValVec[k] = 1 - (M / numObs)
# 		vBarOld = vBar
# 	end
# 	return(pValVec)
# end
# #Non-exported function used to update the \bar{V}_k and \bar{V}_{k,i}^* in White (2000) (see page 1110)
# function rc_updateboot!{T<:Number}(lD::Matrix{T}, inds::Matrix{Int}, rootNumObs::Float64, modelNum::Int, fBar::Float64, fBoot::Vector{Float64}, vBoot::Vector{Float64})
# 	lDSub = sub(lD, :, modelNum)
# 	for i = 1:size(inds, 2)
# 		fBoot[i] = mean(lDSub[sub(inds, :, i)])
# 		vBoot[i] = max(vBoot[i], rootNumObs * (fBoot[i] - fBar))
# 	end
# 	return(true)
# end






