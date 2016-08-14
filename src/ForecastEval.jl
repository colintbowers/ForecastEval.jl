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
		SPABootstrap,
		mcs,
		MCSMethod,
		MCSBootstrap



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
	return(pVal, tailRegion, m)
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
	return(pVal, tailRegion, m)
end
#Diebold-Mariano test using a dependent bootstrap (the forceLocationEstimator keyword arg forces the routine to use the statistic function in the input BootstrapParam, EVEN IF IT IS NOT mean)
function dm{T<:Number}(lossDiff::Vector{T}, method::DMBootstrap; confLevel::Number=0.05, forceLocationEstimator::Bool=false)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	getblocklength(method.bootstrapParam) <= 0 && dbootstrapblocklength!(method.bootstrapParam, lossDiff) #detect block-length if necessary
	if forceLocationEstimator == true
		update!(method.bootstrapParam, numObsData=length(lossDiff), numObsResample=length(lossDiff)) #Ensure correct fixed values in BootstrapParam
		testStat = method.bootstrapParam.statistic(lossDiff)
	else
		update!(method.bootstrapParam, numObsData=length(lossDiff), numObsResample=length(lossDiff), statistic=mean) #Ensure correct fixed values in BootstrapParam
		testStat = mean(lossDiff)
	end
	statVec = dbootstrapstatistic(lossDiff, method.bootstrapParam)
	statVec -= mean(statVec) #Centre statVec on zero (not affected by forceLocationEstimator)
	sort!(statVec)
	(pVal, tailRegion) = pvalue(statVec, testStat)
	pVal > confLevel && (tailRegion = 0)
	return(pVal, tailRegion, mean(lossDiff))
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
	T != Float64 && (lD = Float64[ Float64(lD[j, k]) for j = 1:size(lD, 1), k = 1:size(lD, 2) ]) #Ensure input is Float64
	#Hansen's loss differentials have base case first
	lD *= -one(T)
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
	return(pValVec) #For :all, order is [u, c, l]
end
#Keyword wrapper
function spa{T<:Number}(lD::Matrix{T}; method::SPAMethod=SPABootstrap(size(lD, 1)), numResample::Int=-999, blockLength::Number=-999, blockLengthFilter::Symbol=:none, hacVariant::Symbol=:none, muMethod::Symbol=:none)
	update!(method.bootstrapParam, numResample=numResample, blockLength=blockLength)
	update!(method, size(lD, 1), hacVariant=hacVariant, muMethod=muMethod, blockLengthFilter=blockLengthFilter)
	return(spa(lD, method))
end
#Keyword wrapper (calculates loss differential)
spa{T<:Number}(xPrediction::Matrix{T}, xBaseCase::Vector{T}, xTrue::Vector{T}; lossFunction::LossFunction=SquaredLoss(), method::SPAMethod=SPABootstrap(size(lD, 1)), numResample::Int=-999, blockLength::Number=-999, blockLengthFilter::Symbol=:none, hacVariant::Symbol=:none, muMethod::Symbol=:none) = spa(lossdiff(xPrediction, xBaseCase, xTrue, lossFunction), method=method, numResample=numResample, blockLength=blockLength, blockLengthFilter=blockLengthFilter, hacVariant=hacVariant, muMethod=muMethod)





#----------------------------------------------------------
#ROUTINE
#	Model Confidence Set
#OUTPUT
#REFERENCES
#	Hansen, Lunde, Nason (2011) "The Model Confidence Set", Econometrica, 79 (2), pp. 453-497
#NOTES
#	You can skip the block length selection procedure by specifying a block-length > 0
#----------------------------------------------------------
#Type definitions
abstract MCSMethod <: ForecastEvalMethodNoBaseCase
type MCSBootstrap <: MCSMethod #Method type for performing MCS
	bp::BootstrapParam
	blockLengthFilter::Symbol
	function MCSBootstrap(bp::BootstrapParam, blockLengthFilter::Symbol)
		!(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) && error("Invalid value for blockLengthFilter")
		new(bp, blockLengthFilter)
	end
end
MCSBootstrap(bp::BootstrapParam) = MCSBootstrap(bp, :median)
MCSBootstrap(numObs::Int ; blockLength::Float64=-1.0, blockLengthFilter::Symbol=:median) = MCSBootstrap(BootstrapParam(numObs, blockLength=blockLength), blockLengthFilter)
MCSBootstrap{T<:Number}(l::AbstractMatrix{T} ; blockLength::Float64=-1.0, blockLengthFilter::Symbol=:median) = MCSBootstrap(size(l, 1), blockLength=blockLength, blockLengthFilter=blockLengthFilter)
type MCSOut #Output from MCS
	inA::Vector{Int}
	outA::Vector{Int}
	pValA::Vector{Float64}
	inB::Vector{Int}
	outB::Vector{Int}
	pValB::Vector{Float64}
	MCSOut(inA::Vector{Int}, outA::Vector{Int}, pValA::Vector{Float64},	inB::Vector{Int}, outB::Vector{Int}, pValB::Vector{Float64}) = (inA, outA, pValA, inB, outB, pValB)
end
#Type methods
Base.string(x::MCSBootstrap) = "mcsBootstrap"
Base.string(x::MCSOut) = "mcsOut"
function Base.show(io::IO, x::MCSBootstrap)
	println(string(x) * " contents:")
	show(io, x.bp)
	println("Block length filter = " * string(x.blockLengthFilter))
end
Base.show(x::MCSBootstrap) = show(STDOUT, x)
function Base.show(io::IO, x::MCSOut)
	println("Model confidence set output:")
	if length(x.pValA) == length(x.pValB) == 0
		println("Object is empty")
	else
		length(x.pValA) > 0 && println("Models in MCS method A = " * string(x.inA))
		length(x.pValB) > 0 && println("Models in MCS method B = " * string(x.inB))
	end
end
Base.show(x::MCSOut) = show(STDOUT, x)
function update!(x::MCSBootstrap ; blockLengthFilter::Symbol=:none)
	if blockLengthFilter != :none
		(blockLengthFilter == :mean || blockLengthFilter == :median || blockLengthFilter == :maximum || blockLengthFilter == :first) ? (x.blockLengthFilter = blockLengthFilter) : error("Invalid symbol for blockLengthFilter field")
	end
	return(x)
end
#Function for implementing the Model Confidence Set (MCS). Input l is a matrix of losses
#ISSUE 1: Some of the temporary arrays in the loops could probably be eliminated
#ISSUE 2: For MCS method A, I think the loop over K could be terminated as soon as cumulative p-values are greater than confLevel. Need to double check this.
#ISSUE 3: Need to add option to do just max(abs) method or just sum(sq) method (or both)
function mcs{T<:Number}(l::Matrix{T}, method::MCSBootstrap ; confLevel::Float64=0.05)
	#Validate inputs
	!(0.0 < confLevel < 1.0) && error("Invalid confidence level")
	(N, K) = size(l)
	N < 2 && error("Input must have at least two observations")
	K < 2 && error("Input must have at least two models")
	T != Float64 && (l = Float64[ Float64(l[j, k]) for j = 1:size(l, 1), k = 1:size(l, 2) ]) #Convert input to Float64
	#Get block-length if necessary and then get bootstrap indices. Note, appropriate block length is estimated by examining loss differentials between models 2 to K and model 1. Checking block length on every j, k combination will take too long for large K
	getblocklength(method.bp) <= 0.0 && multivariate_blocklength!(Float64[ l[n, k] - l[n, 1] for n = 1:N, k = 2:K ], method.bp, method.blockLengthFilter)
	inds = dbootstrapindex(method.bp)
	#Get matrix of loss differential sample means
	lMuVec = mean(l, 1)
	lDMu = Float64[ lMuVec[k] - lMuVec[j] for j = 1:K, k = 1:K  ]
	#Get array of  bootstrapped loss differential sample means
	lDMuStar = Array(Float64, K, K, method.bp.numResample) #This array is affected by ISSUE 1 above
	for m = 1:method.bp.numResample
		lMuVecStar = mean(l[inds[:, m], :], 1)
		lDMuStar[:, :, m] = Float64[ lMuVecStar[k] - lMuVecStar[j] for j = 1:K, k = 1:K  ]
	end
	#Get variance estimates from bootstrapped loss differential sample means (note, we centre on lDMu since these are the population means for the resampled data)
	#Note, for efficiency, we only use varm to fill out the lower triangular matrix
	lDMuVar = ones(Float64, K, K)
	for j = 2:K
		for k = 1:j-1
			lDMuVar[j, k] = varm(vec(lDMuStar[j, k, :]), lDMu[j, k], corrected=false)
		end
	end
	ltri_to_utri!(lDMuVar)
	#Get original and re-sampled t-statistics
	tStatStar = Float64[ (lDMuStar[j, k, m] - lDMu[j, k]) / sqrt(lDMuVar[j, k]) for j = 1:K, k = 1:K, m = 1:method.bp.numResample ]
	tStat = Float64[ lDMu[j, k] / sqrt(lDMuVar[j, k]) for j = 1:K, k = 1:K ]
	#Perform model confidence method A
	inA = collect(1:K) #Models in MCS (start off with all models included)
	outA = Array(Int, K) #Models not in MCS (start off with no models in MCS)
	pValA = ones(Float64, K) #p-values constructed in loop
	for k = 1:K-1
		bootMax = Float64[ maxabs(tStatStar[inA, inA, m]) for m = 1:method.bp.numResample ]
		#bootMax = vec(maximum(abs(tStatStar[inA, inA, :]), [1, 2])) #ALTERNATIVE METHOD
		origMax = maximum(tStat[inA, inA])
		pValA[k] = mean(bootMax .> origMax)
		scalingTerm = length(inA) / (length(inA) - 1)
		lDAvgMu = scalingTerm * vec(mean(lDMu[inA, inA], 1))
		lDAvgMuStar = scalingTerm * squeeze(mean(lDMuStar[inA, inA, :], 1), 1)
		lDAvgMuVar = Float64[ varm(vec(lDAvgMuStar[k, :]), lDAvgMu[k], corrected=false) for k = 1:length(lDAvgMu) ]
		tStatInc = lDAvgMu ./ sqrt(lDAvgMuVar)
		iRemove = indmax(tStatInc) #Find index in inA of model to be removed
		outA[k] = inA[iRemove] #Add model to be removed to excluded list
		deleteat!(inA, iRemove) #Remove model to be removed
	end
	pValA = cummax(pValA)
	outA[end] = inA[1] #Finish constructing excluded models
	iCutOff = findfirst(pValA .>= confLevel) #confLevel < 1.0, hence there will always be at least one p-value > confLevel
	inA = outA[iCutOff:end]
	outA = outA[1:iCutOff-1]
	#Perform model confidence set method B
	inB = collect(1:K) #Models in MCS (start off with all models in MCS)
	outB = Array(Int, K) #Models not in MCS (start off with no models in MCS)
	pValB = ones(Float64, K) #p-values constructed in loop
	for k = 1:K-1
		bootSum = 0.5 * vec(sumabs2(tStatStar[inB, inB, :], [1, 2]))
		origSum = 0.5 * sumabs2(tStat[inB, inB])
		pValB[k] = mean(bootSum .> origSum)
		scalingTerm = length(inB) / (length(inB) - 1)
		lDAvgMu = scalingTerm * vec(mean(lDMu[inB, inB], 1))
		lDAvgMuStar = scalingTerm * squeeze(mean(lDMuStar[inB, inB, :], 1), 1)
		lDAvgMuVar = Float64[ varm(vec(lDAvgMuStar[k, :]), lDAvgMu[k], corrected=false) for k = 1:length(lDAvgMu) ]
		tStatInc = lDAvgMu ./ sqrt(lDAvgMuVar)
		iRemove = indmax(tStatInc) #Find index in inB of model to be removed
		outB[k] = inB[iRemove] #Add model to be removed to excluded list
		deleteat!(inB, iRemove) #Remove model to be removed
	end
	pValB = cummax(pValB)
	outB[end] = inB[1] #Finish constructing excluded models
	iCutOff = findfirst(pValB .>= confLevel) #confLevel < 1.0, hence there will always be at least one p-value > confLevel
	inB = outB[iCutOff:end]
	outB = outB[1:iCutOff-1]
	#Prepare the output
	mcsOut = MCSOut(inA, outA, pValA, inB, outB, pValB)
end
function mcs_optim{T<:Number}(l::Matrix{T}, method::MCSBootstrap ; confLevel::Float64=0.05)
	#Validate inputs
	!(0.0 < confLevel < 1.0) && error("Invalid confidence level")
	(N, K) = size(l)
	N < 2 && error("Input must have at least two observations")
	K < 2 && error("Input must have at least two models")
	T != Float64 && (l = Float64[ Float64(l[j, k]) for j = 1:size(l, 1), k = 1:size(l, 2) ]) #Convert input to Float64
	#Get block-length if necessary and then get bootstrap indices. Note, appropriate block length is estimated by examining loss differentials between models 2 to K and model 1. Checking block length on every j, k combination will take too long for large K
	getblocklength(method.bp) <= 0.0 && multivariate_blocklength!(Float64[ l[n, k] - l[n, 1] for n = 1:N, k = 2:K ], method.bp, method.blockLengthFilter)
	inds = dbootstrapindex(method.bp)
	#Get matrix of loss differential sample means
	lMuVec = vec(mean(l, 1))
	iM = ltri_cart_index(collect(1:K)) #Build a matrix of lower triangular cartesian indices
	S = size(iM, 1) #Total number of cross series
	lDMuCross = Float64[ lMuVec[iM[s, 2]] - lMuVec[iM[s, 1]] for s = 1:S ] #diag = 0.0, utri = -1.0*ltri
	#Get array of  bootstrapped loss differential sample means
	lDMuCrossStar = Array(Float64, S, method.bp.numResample)
	for m = 1:method.bp.numResample
		lMuVecStar = mean(l[inds[:, m], :], 1)
		lDMuCrossStar[:, m] = Float64[ lMuVecStar[iM[s, 2]] - lMuVecStar[iM[s, 1]] for s = 1:S ] #diag = 0.0, utri = -1.0*ltri
	end
	#Get variance estimates from bootstrapped loss differential sample means (note, we centre on lDMu since these are the population means for the resampled data)
	lDMuCrossVar = Float64[ varm(lDMuCrossStar[s, :], lDMuCross[s], corrected=false) for s = 1:S ] #diag = 1.0, utri = ltri
	#Get original and re-sampled t-statistics
	tStatCrossStar = Float64[ (lDMuCrossStar[s, m] - lDMuCross[s]) / sqrt(lDMuCrossVar[s]) for s = 1:S, m = 1:method.bp.numResample ] #diag = 0.0, utri = -1.0*ltri
	tStatCross = Float64[ lDMuCross[s] / sqrt(lDMuCrossVar[s]) for s = 1:S ] #diag = 0.0, utri = -1.0*ltri
	#Perform model confidence method A
	inA = collect(1:K) #Models in MCS (start off with all models included)
	outA = Array(Int, K) #Models not in MCS (start off with no models in MCS)
	pValA = ones(Float64, K) #p-values constructed in loop
	for k = 1:K-1
		iIn = ltri_index_match(K, inA) #Linear indices of models that are still in the MCS
		bootMaxIn = Float64[ maxabs(tStatCrossStar[iIn, m]) for m = 1:method.bp.numResample ]
		origMaxIn = maxabs(tStatCross[iIn])
		pValA[k] = mean(bootMaxIn .> origMaxIn)
		lDAvgMuCross = vec(mean(msym_mat_from_ltri_inds(lDMuCross, iIn), 1))
		lDAvgMuCrossStar = Array(Float64, method.bp.numResample, trinumroot(length(iIn))+1)
		for s = 1:method.bp.numResample
			lDAvgMuCrossStar[s, :] = mean(msym_mat_from_ltri_inds(lDMuCrossStar[:, s], iIn), 1)
		end
		lDAvgMuCrossVar = Float64[ varm(lDAvgMuCrossStar[:, k], lDAvgMuCross[k], corrected=false) for k = 1:length(lDAvgMuCross) ]
		tStatCrossInc = lDAvgMuCross ./ sqrt(lDAvgMuCrossVar)
		iRemove = indmax(tStatCrossInc) #Find index in inA of model to be removed
		outA[k] = inA[iRemove] #Add model to be removed to excluded list
		deleteat!(inA, iRemove) #Remove model to be removed
	end
	pValA = cummax(pValA)
	outA[end] = inA[1] #Finish constructing excluded models
	iCutOff = findfirst(pValA .>= confLevel) #confLevel < 1.0, hence there will always be at least one p-value > confLevel
	inA = outA[iCutOff:end]
	outA = outA[1:iCutOff-1]
	#Perform model confidence set method B
	inB = collect(1:K) #Models in MCS (start off with all models in MCS)
	outB = Array(Int, K) #Models not in MCS (start off with no models in MCS)
	pValB = ones(Float64, K) #p-values constructed in loop
	for k = 1:K-1
		iIn = ltri_index_match(K, inB) #Linear indices of models that are still in the MCS
		bootSumIn = vec(sumabs2(tStatCrossStar[iIn, :], 1))
		origSumIn = sumabs2(tStatCross[iIn])
		pValB[k] = mean(bootSumIn .> origSumIn)
		lDAvgMuCross = vec(mean(msym_mat_from_ltri_inds(lDMuCross, iIn), 1))
		lDAvgMuCrossStar = Array(Float64, method.bp.numResample, trinumroot(length(iIn))+1)
		for s = 1:method.bp.numResample
			lDAvgMuCrossStar[s, :] = mean(msym_mat_from_ltri_inds(lDMuCrossStar[:, s], iIn), 1)
		end
		lDAvgMuCrossVar = Float64[ varm(lDAvgMuCrossStar[:, k], lDAvgMuCross[k], corrected=false) for k = 1:length(lDAvgMuCross) ]
		tStatCrossInc = lDAvgMuCross ./ sqrt(lDAvgMuCrossVar)
		iRemove = indmax(tStatCrossInc) #Find index in inB of model to be removed
		outB[k] = inB[iRemove] #Add model to be removed to excluded list
		deleteat!(inB, iRemove) #Remove model to be removed
	end
	pValB = cummax(pValB)
	outB[end] = inB[1] #Finish constructing excluded models
	iCutOff = findfirst(pValB .>= confLevel) #confLevel < 1.0, hence there will always be at least one p-value > confLevel
	inB = outB[iCutOff:end]
	outB = outB[1:iCutOff-1]
	#Prepare the output
	mcsOut = MCSOut(inA, outA, pValA, inB, outB, pValB)
end
#Local function to shift lower triangular elements to upper triangular portion of matrix
function ltri_to_utri!{T<:Number}(x::AbstractMatrix{T})
	size(x, 1) != size(x, 2) && error("Input matrix  must be symmetric")
	for j = 2:size(x, 1)
		for k = 1:j-1
			x[k, j] = x[j, k]
		end
	end
	return(true)
end
#Local function for triangular numbers
trinum(K::Int) = Int((K*(K+1))/2)
trinumroot(triNum::Int) = Int((sqrt(8*triNum + 1) - 1) / 2)
#Local functions for matching lower triangular cartesian indices to a linear index
ltri_index_match(K::Int, i::Int, j::Int) = trinum(K-1) - trinum(K-j-1) - K + i
function ltri_index_match(K::Int, cartInds::Matrix{Int})
	size(cartInds, 2) != 2 && error("Invalid cartesian index matrix")
	tnK = trinum(K-1)
	return(Int[ tnK - trinum(K - cartInds[s, 2] - 1) - K + cartInds[s, 1] for s = 1:size(cartInds, 1) ])
end
ltri_index_match(K::Int, inds::Vector{Int}) = ltri_index_match(K, ltri_cart_index(inds))
#Local function for constructing a matrix of lower triangular cartesian indices
function ltri_cart_index(inds::Vector{Int})
	indsOut = Array(Int, trinum(length(inds)-1), 2)
	c = 1
	for k = 1:length(inds)-1
		for j = k+1:length(inds)
			indsOut[c, 1] = inds[j]
			indsOut[c, 2] = inds[k]
			c += 1
		end
	end
	return(indsOut)
end
#Local function for constructing a matrix from a lower triangle using linear indices (no diagonal), and minus one times the lower triangle (no diagonal). Two components are stuck together transposed to each other.
function msym_mat_from_ltri_inds{T<:Number}(x::AbstractVector{T}, linInds::Vector{Int})
	K = trinumroot(length(linInds))
	xOut = Array(T, K, K+1)
	iSt = 1
	for k = K:-1:1
		triCol = x[linInds[iSt:iSt+k-1]]
		xOut[K-k+1:end, K-k+1] = triCol
		xOut[K-k+1, K-k+2:end] = -1.0 * triCol
		iSt += k
	end
	return(xOut)
end
#Keyword wrapper
function mcs{T<:Number}(l::Matrix{T}; method::MCSMethod=MCSBootstrap(l), numResample::Int=1000, blockLength::Number=-1.0, blockLengthFilter::Symbol=:median, confLevel::Float64=0.05)
	update!(method.bp, numResample=numResample, blockLength=blockLength)
	update!(method, blockLengthFilter=blockLengthFilter)
	return(mcs(l, method, confLevel=confLevel))
end








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
	i = searchsortedlast(xVec, xObs)
	if twoSided
		if xObs <= mean(xVec)
			tailRegion = -1
			pv = 2*(i/length(xVec))
		else
			tailRegion = 1
			pv = 2*(1 - (i/length(xVec)))
		end
	else
		tailRegion = 1
		pv = 1 - (i/length(xVec))
	end
	return(pv, tailRegion)
end
function pvalue{T<:Number}(d::Distribution, xObs::T; twoSided::Bool=true)
	if twoSided
		if xObs < mean(d)
			tailRegion = -1
			pv = 2*cdf(d, xObs)
		else
			tailRegion = 1
			pv = 2*(1 - cdf(d, xObs))
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
