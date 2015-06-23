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
export 	dieboldMariano,
		DMMethod,
		DMAsymptotic,
		DMBootstrap


#******************************************************************************

#----------------------------------------------------------
#SET CONSTANTS FOR MODULE
#----------------------------------------------------------





#----------------------------------------------------------
#TYPE
#	DMStandard
#	DMBootstrap
#FIELDS
#	Fields store parameteres necessary to perform a Diebold-Mariano test using the indicated method
#METHODS
#PURPOSE
#	These types are used to call different methods of dieboldMariano, depending on how the user wants to perform the test.
#NOTES
#----------------------------------------------------------
#Abstract type for nesting method types for diebold mariano test
abstract DMMethod
#Type indicating to do the asymptotic test (section 1.1 of their paper)
type DMAsymptotic <: DMMethod
	lagWindow::Int
	function DMAsymptotic(lagWindow::Int)
		lagWindow < 0 && error("Lag window must be non-negative")
		new(lagWindow)
	end
end
DMAsymptotic() = DMAsymptotic(0)
#Type indicating to use a bootstrap to conduct the test
type DMBootstrap <: DMMethod
	bootstrapParam::BootstrapParam
	function DMBootstrap(bootstrapParam::BootstrapParam)
		if bootstrapParam.statistic != "mean" && bootstrapParam.statistic != mean
			error("statistic field in BootstrapParam must be set to mean")
		end
		new(bootstrapParam)
	end
end
DMBootstrap{T<:Number}(lossDiff::Vector{T}) = DMBootstrap(BootstrapParam(lossDiff, bootstrapMethod="stationary", statistic="mean"))
#------- METHODS --------------
#string methods
Base.string(d::DMAsymptotic) = "dmAsymptotic"
Base.string(d::DMBootstrap) = "dmBootstrap"
#show methods
function Base.show(io::IO, d::DMAsymptotic)
	println(io, "Diebold-Mariano Asymptotic Method:")
	pritnln(io, "    Lag window = " * string(d.lagWindow))
end
function Base.show(io::IO, d::DMBootstrap)
	println(io, "Diebold-Mariano Bootstrap Method:")
	show(io, d.bootstrapParam)
end
#show wrapper on STDOUT
Base.show(d::DMMethod) = show(STDOut, d)
#----------------------------------------------------------
#FUNCTION
#	dieboldMariano
#INPUT
#OUTPUT
#PURPOSE
#	Perform a Diebold-Mariano (1995) test for determining if there is a difference in loss between two forecasting models
#NOTES
#REFERENCES
#	Diebold, Francis X., Mariano, Roberto S. (1995) "Comparing Predictive Accuracy", Journal of Business and Economic Statistics, 13 (3), pp. 253-263
#----------------------------------------------------------
#Diebold-Mariano test using asymptotic approximation (see section 1.1 of their paper)
function dieboldMariano{T<:Number}(lossDiff::Vector{T}, method::DMAsymptotic; confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	m = mean(lossDiff)
	if method.lagWindow == 0
		v = var(lossDiff)
	else
		v = var(lossDiff) + 2 * sum(autocov(lossDiff, 1:method.lagWindow)) #Rectangular kernel to specified lag
	end
	v <= 0 && v = 0 #If variance estimate is negative, bind to zero so we auto-reject the null
	testStat = m / sqrt(v / length(lossDiff))
	testStat < 0 ? (pVal = 2*cdf(Normal(), testStat)) : (pVal = 2*cdf(Normal(), -1*testStat)) #Get p-value
	pVal < confLevel ? return(sign(m), pVal) : return(0, pVal) #Sign of output flag tells you which forecast was preferred
end
#Diebold-Mariano test using a dependent bootstrap
function dieboldMariano{T<:Number}(lossDiff::Vector{T}, method::DMBootstrap; confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	getBlockLength(method.bootstrapParam) <= 0 && dBootstrapBlockLength!(method.bootstrapParam, lossDiff) #detect block-length if necessary
	method.bootstrapParam.statistic != "mean" && error("statistic field in BootstrapParam must be set to mean")
	statVec = dBootstrapStatistic(lossDiff, method.bootstrapParam)
	sort!(statVec)
	i = searchsortedlast(statVec, 0.0)
	NHalf = 0.5 * length(statVec)
	if i < NHalf
		pVal = i / NHalf
		outSign = 1
	else
		pVal = (length(statVec) - i) / NHalf
		outSign = -1
	end
	pVal < confLevel ? return(outSign, pVal) : return(0, pVal)
end
#Keyword method (loss differential already computed)
dieboldMariano{T<:Number}(lossDiff::Vector{T}; method::ASCIIString="bootstrap", numResample::Int=1000, lagWindow::Int=0, confLevel::Number=0.05) = dieboldMariano(lossDiff, dM_getmethod(length(lossDiff), method, numResample, lagWindow), confLevel=confLevel)
#Wrapper (loss differential not computed)
dieboldMariano{T<:Number}(xhat1::Vector{T}, xhat2::Vector{T}, x::Vector{T}, lossFunc::LossFunction, method::DMMethod, confLevel::Number=0.05) = dieboldMariano(dM_getlossdiff(xhat1, xhat2, x, lossFunc), method, confLevel=confLevel)
#Keyword method wrapper (loss differential not computed)
dieboldMariano{T1<:Number, T2<:Union(ASCIIString, LossFunction)}(xhat1::Vector{T1}, xhat2::Vector{T1}, x::Vector{T1}; lossFunc::T2=SquaredError(), method::ASCIIString="bootstrap", numResample::Int=1000, lagWindow::Int=0, confLevel::Number=0.05) = dieboldMariano(dM_getlossdiff(xhat1, xhat2, x, lossFunc), dM_getmethod(length(lossDiff), method, numResample, lagWindow), confLevel=confLevel)
#Non-exported functions that help with keyword wrapper routines
function dM_getmethod(numObs::Int, method::ASCIIString, numResample::Int, lagWindow::Int)
	if method == "asymptotic"
		methodType = DMAsymptotic(lagWindow)
	elseif method == "bootstrap" || method == "stationary" || method == "circularBlock" || method == "movingBlock"
		bp = BootstrapParam(numObs, bootstrapMethod=method, numResample=numResample, statistic="mean")
		methodType = DMBootstrap(bp)
	else
		error("Unable to infer desired method from inputs")
	end
	return(methodType)
end
function dM_getlossdiff{T<:Number}(xhat1::Vector{T}, xhat2::Vector{T}, x::Vector{T}, lossFunc::LossFunction)
	!(length(xhat1) == length(xhat2) == length(x)) && error("Input data vectors must all be of equal length")
	return(loss(xhat1, x, lossFunc) - loss(xhat2, x, lossFunc))
end
function dM_getlossdiff{T<:Number}(xhat1::Vector{T}, xhat2::Vector{T}, x::Vector{T}, lossFunc::ASCIIString)
	if lossFunc = "MSE"
		lossFunc = SquaredError()
	elseif lossFunc = "MAE"
		lossFunc = AbsoluteError()
	elseif lossFunc = "QLIKE"
		lossFunc = QLIKE()
	else
		error("lossFunc string not recognised")
	end
	return(dieboldMariano_getlossdiff(xhat1, xhat2, x, lossFunc))
end



#----------------------------------------------------------
#TYPE
#	RCBootstrap
#FIELDS
#	Fields store parameteres necessary to perform a Reality-Check test using the indicated method
#METHODS
#PURPOSE
#	These types are used to call different methods of realityCheck, depending on how the user wants to perform the test.
#NOTES
#----------------------------------------------------------
type RCBootstrap
	bootstrapParam::BootstrapParam
	blockLengthFilter::ASCIIString
	function RCBootstrap(bootstrapParam::BootstrapParam, blockLengthFilter::ASCIIString)
		!(blockLengthFilter == "mean" || blockLengthFilter == "median" || blockLengthFilter == "maximum" || blockLengthFilter == "first") && error("Invalid value for blockLengthFilter")
		new(bootstrapParam, blockLengthFilter)
	end
end
Base.string(x::RCBootstrap) = "rcBootstrap"
function Base.show(io::IO, x::RCBootstrap)
	println(io, "Reality check via bootstrap. Bootstrap parameters are:")
	show(io, x.bootstrapParam)
end
Base.show(x::RCBootstrap) = show(STDOUT, x)
#----------------------------------------------------------
#FUNCTION
#	realityCheck
#INPUT
#OUTPUT
#PURPOSE
#	Perform a White (2000) reality check for determining if a set of forecast models outperform a benchmark
#NOTES
#REFERENCES
#	White (2000) "A Reality Check for Data Snooping", Econometrica, 68 (5), pp. 1097-1126
#----------------------------------------------------------
function realityCheck{T<:Number}(lD::Matrix{T}, method::RCBootstrap)
	numObs = size(lD, 1)
	numModel = size(lD, 2)
	rootNumObs = sqrt(numObs)
	numObs < 3 && error("Not enough observations to perform a reality check")
	numModel < 1 && error("Input data matrix is empty")
	realityCheck_blocklength!(lD, method) #Gets appropriate block-length (if needed)
	method.bootstraParam.numObsData != numObs && error("numObsData field in BootstrapParam is not equal to number of rows in loss differential matrix")
	typeof(method.bootstrapParam.bootstrapMethod) == TaperedBlock && error("realityCheck currently not possible using tapered block bootstrap")
	inds = dBootstrapIndex(method.bootstrapParam)
	fBoot = Array(Float64, size(inds, 2))
	vBoot = Array(Float64, size(inds, 2))
	vBoot *= -Inf #All -Inf values will get updated on first iteration
	pValVec = Array(Float64, numModel)
	vBarOld = -Inf
	for k = 1:numModel
		fBar = mean(sub(lD, 1:numObs, k))
		vBar = max(vBarOld, rootNumObs * fBar)
		realityCheck_updateboot!(lD, inds, numObs, rootNumObs, k, fBar, fBoot, vBoot)
		#DO TEST AND GET PVALUE HERE
	end

	#FINISH UP HERE
end
function realityCheck_updateboot!{T<:Number}(lD::Matrix{T}, inds::Matrix{Int}, numObs::Int, rootNumObs::Float64, modelNum::Int, fBar::Float64, fBoot::Vector{Float64}, vBoot::Vector{Float64})
	lDSub = sub(lD, 1:numObs, modelNum)
	for i = 1:size(inds, 2)
		fBoot[i] = mean(lDSub[sub(inds, 1:numObs, i)])
		vBoot[i] = max(vBoot[i], rootNumObs * (fBoot[i] - fBar))
	end
	return(true)
end




function realityCheck_blocklength!{T<:Number}(lD::Matrix{T}, method::RCBootstrap)
	if getBlockLength(method.bootstrapParam) <= 0
		if blockLengthFilter == "first"
			bL = dBootstrapBlockLength(lD[:, 1], method.bootstrapParam)
		else
			blockLengthVec = [ dBootstrapBlockLength(lD[:, n], method.bootstrapParam) for n = 1:size(lD, 2) ]
			if blockLengthFilter == "mean"
				bL = mean(blockLengthVec)
			elseif blockLengthFilter == "median"
				bL = median(blockLengthVec)
			elseif blockLengthFilter == "maximum"
				bL = max(blockLengthVec)
			else
				error("Invalid value for blockLengthFilter")
			end
		end
		replaceBlockLength!(method.bootstrapParam, bL)
	else
		bL = convert(Float64, getBlockLength(method.bootstrapParam))
	end
	return(bL)
end



end # module




