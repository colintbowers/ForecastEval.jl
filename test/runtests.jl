using ForecastEval
using KernelStat
using ARIMA
using DependentBootstrap
using LossFunctions

const numObs = 500::Int
const numModel = 10::Int
const dmMethodList = [DMAsymptoticBasic(numObs), DMAsymptoticHAC(numObs), DMBootstrap(numObs)]::Vector{DMMethod}
const rcMethodList = [RCBootstrap(numObs), RCBootstrapAlt(numObs)]::Vector{RCMethod}
const spaMethodList = [SPABootstrap(numObs), SPABootstrap(numObs, :epanechnikov)]::Vector{SPABootstrap}


function simmodel(maOrder::Int, eVarMin::Float64=0.1, eVarMax::Float64=0.25)
	maOrder == 0 ? maCoef = Array(Float64, 0) : maCoef = [ 1 / 2^p for p = 1:maOrder ]
	xTrue = simulate(numObs, ARIMAModel(maCoef=maCoef))
	vbc = eVarMin + (eVarMax - eVarMin) * rand()
	vModel = eVarMin + (eVarMax - eVarMin) * rand(numModel)
	bc = xTrue + vbc * randn(numObs)
	x = [ xTrue + vModel[n] * randn(numObs) for n = 1:numModel ]
	ranking = sortperm([vModel, vbc])
	ranking[ranking .== numModel+1] = 0
	return(xTrue, vbc, bc, vModel, x, ranking)
end
col2mat{T}(x::Vector{Vector{T}}) = [ x[m][n] for n = 1:length(x[1]), m = 1:length(x) ]
mat2col{T}(x::Matrix{T}) = [ x[:, m] for m = 1:size(x, 2) ]




function testdm()
	for maOrder = 0:3:6
		(xTrue, vbc, bc, vModel, x, ranking) = simmodel(maOrder, 0.2, 0.33)
		for j = 1:length(dmMethodList)
			dmM = dmMethodList[j]
			pValVec = Array(Float64, numModel)
			tailRegionVec = Array(Int, numModel)
			for k = 1:numModel
				(pValVec[k], tailRegionVec[k]) = dm(x[k], bc, xTrue, method=dmM)
			end
			show(dmM)
			println("True MA order = " * string(maOrder))
			println("True ranking = " * string(ranking))
			println("Base-case error variance = " * string(vbc))
			println("Model error variances = " * string(vModel))
			println("dm p-values = " * string(pValVec))
			println("dm tail regions = " * string(tailRegionVec))
			println("")
			println("")
		end
	end
	return(true)
end



function testrc()
	for maOrder = 0:3:6
		(xTrue, vbc, bc, vModel, x, ranking) = simmodel(maOrder)
		x = col2mat(x)
		for j = 1:length(rcMethodList)
			rcM = rcMethodList[j]
			pValVec = rc(x, bc, xTrue, method=rcM)
			println("True MA order = " * string(maOrder))
			println("True ranking = " * string(ranking))
			println("Base-case error variance = " * string(vbc))
			println("Model error variances = " * string(vModel))
			println("rc p-values = " * string(pValVec))
			println("")
			println("")
		end
	end
	return(true)
end


testrc()


function testspa()
	for maOrder = 0:3:6
		(xTrue, vbc, bc, vModel, x, ranking) = simmodel(maOrder)
		x = [ x[m][n] for n = 1:numObs, m = 1:numModel ]
		for j = 1:length(spaMethodList)
			spaM = spaMethodList[j]
			pValVec = spa(x, bc, xTrue, method=spaM)
			println("True MA order = " * string(maOrder))
			println("True ranking = " * string(ranking))
			println("Base-case error variance = " * string(vbc))
			println("Model error variances = " * string(vModel))
			println("spa pval_u = " * string(pValVec[1]))
			println("spa pval_l = " * string(pValVec[2]))
			println("spa pval_c = " * string(pValVec[3]))
			println("")
			println("")
		end
	end
	return(true)
end






