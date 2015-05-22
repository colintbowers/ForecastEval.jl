module ForecastEval

#TO ADD
#p-value for bootstrap input to diebold-mariano after building SortedVector type


#-----------------------------------------------------------
#PURPOSE
#	Colin T. Bowers module for forecast evaluation
#NOTES
#LICENSE
#	MIT License (see github repository for more detail: https://github.com/colintbowers/DependentBootstrap.jl.git)
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



#******************************************************************************

#----------------------------------------------------------
#SET CONSTANTS FOR MODULE
#----------------------------------------------------------





#----------------------------------------------------------
#TYPE
#	DMStandard
#	DMOLS
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
string(d::DMAsymptotic) = "dmAsymptotic"
string(d::DMBootstrap) = "dmBootstrap"
#show methods
function show(io::IO, d::DMAsymptotic)
	println(io, "Diebold-Mariano Asymptotic Method:")
	pritnln(io, "    Lag window = " * string(d.lagWindow))
end
function show(io::IO, d::DMBootstrap)
	println(io, "Diebold-Mariano Bootstrap Method:")
	show(io, d.bootstrapParam)
end
#show wrapper on STDOUT
show(d::DMMethod) = show(STDOut, d)



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
function dieboldMariano{T<:Number}(lossDiff::Vector{T}, method::DMAsymptotic, confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	m = mean(lossDiff)
	method.lagWindow == 0 ? v = var(lossDiff) : v = var(lossDiff) + 2 * sum(autocov(lossDiff, 1:method.lagWindow)) #Rectangular kernel to specified lag
	v <= 0 ? v = 0 #If variance estimate is negative, bind to zero so we auto-reject the null
	testStat = m / sqrt(v / length(lossDiff))
	testStat < 0 ? (pVal = 2*cdf(Normal(), testStat)) : (pVal = 2*cdf(Normal(), -1*testStat)) #Get p-value
	pVal < confLevel ? return(sign(m), pVal) : return(0, pVal) #Sign of output flag tells you which forecast was preferred
end
#Diebold-Mariano test using a dependent bootstrap
function dieboldMariano{T<:Number}(lossDiff::Vector{T}, method::DMBootstrap, confLevel::Number=0.05)
	!(0.0 < confLevel < 1.0) && error("Confidence level must lie between 0 and 1")
	length(lossDiff) < 2 && error("Loss differential series must contain at least two observations")
	testStatVec = dBootstrapStatistic(lossDiff, method.bootstrapParam)
	boundVec = quantile(testStatVec, [confLevel/2, 1 - confLevel/2])
	(boundVec[1] < 0 < boundVec[2]) ? return(0, NaN) : return(sign(boundVec[1]), NaN) #If 0 does not lie in bounds, then both bounds will have the same sign
end
#Keyword method
function dieboldMariano{T<:Number}(lossDiff::Vector{T}; method::ASCIIString="bootstrap", numResample::Int=1000, lagWindow::Int=0, confLevel::Number=0.05)
	if method == "asymptotic"
		return(dieboldMariano(lossDiff, DMAsymptotic(lagWindow)), confLevel)
	else
		method == "bootstrap" && (method = "stationary")
		bp = BootstrapParam(lossDiff, bootstrapMethod=method, numResample=numResample, statistic="mean")
		return(dieBoldMariano(lossDiff, DMBootstrap(bp)))
	end
end
#Diebold-Mariano wrapper for calculating loss differential
function dieboldMariano{T<:Number}(xhat1::Vector{T}, xhat2::Vector{T}, x::Vector{T}, lossFunc::LossFunction, method::DMMethod, confLevel::Number=0.05)
	!(length(xhat1) == length(xhat2) == length(x)) && error("Input data vectors must all be of equal length")
	return(dieboldMariano(loss(xhat1, x, lossFunc) - loss(xhat2, x, lossFunc), method, confLevel))
end
#Keyword method wrapper
function dieboldMariano{T1<:Number, T2<:Union(ASCIIString, LossFunction)}(xhat1::Vector{T1}, xhat2::Vector{T1}, x::Vector{T1}; lossFunc::T2=SquaredError(), method::ASCIIString="bootstrap", numResample::Int=1000, lagWindow::Int=0, confLevel::Number=0.05)
	!(length(xhat1) == length(xhat2) == length(x)) && error("Input data vectors must all be of equal length")
	if typeof(lossFunc) == ASCIIString
		if lossFunc = "MSE"
			lossFunc = SquaredError()
		elseif lossFunc = "MAE"
			lossFunc = AbsoluteError()
		elseif lossFunc = "QLIKE"
			lossFunc = QLIKE()
		else
			error("lossFunc string not recognised")
		end
	end
	lossDiff = loss(xhat1, x, lossFunc) - loss(xhat2, x, lossFunc)
	if method == "asymptotic"
		return(dieboldMariano(lossDiff, DMAsymptotic(lagWindow)), confLevel)
	else
		method == "bootstrap" && (method = "stationary")
		bp = BootstrapParam(lossDiff, bootstrapMethod=method, numResample=numResample, statistic="mean")
		return(dieBoldMariano(lossDiff, DMBootstrap(bp)))
	end
end











end # module




