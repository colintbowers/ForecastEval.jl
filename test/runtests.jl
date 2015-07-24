using ForecastEval
using KernelStat
using ARIMA
using DependentBootstrap
using LossFunctions

const numObs = 500::Int
const numModel = 6::Int
const dmMethodList = [DMAsymptoticBasic(numObs), DMAsymptoticHAC(numObs), DMBootstrap(numObs)]::Vector{DMMethod}
const rcMethodList = [RCBootstrap(numObs), RCBootstrapAlt(numObs)]::Vector{RCMethod}
const spaMethodList = [SPABootstrap(numObs, muMethod=:all), SPABootstrap(numObs, muMethod=:auto), SPABootstrap(numObs, hacVariant=:epanechnikov, muMethod=:auto), SPABootstrap(numObs, blockLength=10, hacVariant=:epanechnikov, muMethod=:auto)]::Vector{SPABootstrap}

function simlD(maOrder::Int, elD::Vector{Float64})
	maOrder == 0 ? maCoef = Array(Float64, 0) : maCoef = [ 1 / 2^p for p = 1:maOrder ]
	lD = Array(Float64, numObs, length(elD))
	for k = 1:length(elD)
		lD[:, k] = simulate(numObs, ARIMAModel(maCoef=maCoef, intercept=elD[k]))
	end
	ranking = flipud(sortperm(elD))
	return(lD, ranking)
end



function testrc(elD::Vector{Float64}=zeros(Float64, numModel))
	numIter = 100
	#for maOrder = 0:0
	for maOrder = 0:3:6
		for j = 2:2
		#for j = 1:length(rcMethodList)
			rcM = rcMethodList[j]
			println("Method = " * string(rcM))
			println("True MA order = " * string(maOrder))
			println("True Expected LD = " * string(elD))
			pValArr = Array(Float64, numIter)
			for n = 1:numIter
				(lD, ranking) = simlD(maOrder, elD)
				rcMIn = deepcopy(rcM)
				pValArr[n] = rc(lD, method=rcMIn)
			end
			println("rc rej freq = " * string((1/numIter) * sum(pValArr .< 0.05)))
			println("")
		end
	end
	return(true)
end









function testspa(elD::Vector{Float64}=zeros(Float64, numModel))
	numIter = 100
	#for maOrder = 0:0
	for maOrder = 0:3:6
		#for j = 1:1
		for j = 1:length(spaMethodList)
			spaM = spaMethodList[j]
			println("Method = " * string(spaM) * "-" * string(spaM.muMethod) * "-" * string(spaM.hacVarianceMethod.kernelFunction))
			println("True MA order = " * string(maOrder))
			println("True Expected LD = " * string(elD))
			pValArr = Array(Vector{Float64}, numIter)
			for n = 1:numIter
				(lD, ranking) = simlD(maOrder, elD)
				spaMIn = deepcopy(spaM)
				pValArr[n] = spa(lD, method=spaMIn)
			end
			numMethod = length(pValArr[1])
			for q = 1:numMethod
				pValVec = Float64[ pValArr[n][q] for n = 1:numIter ]
				println("spa rej freq method " * string(q) * " = " * string((1/numIter) * sum(pValVec .< 0.05)))
			end
			println("")
		end
	end
	return(true)
end




function testdm(elD::Float64=0.0)
	numIter = 100
	if elD < 0.0; trueRanking = 1
	elseif elD == 0.0; trueRanking = 0
	else; trueRanking = -1
	end
#	for maOrder = 0:1:20
	for maOrder = 0:3:6
		for j = 2:length(dmMethodList)
			dmM = dmMethodList[j]
			println("dm Method = " * string(dmM))
			println("True MA order = " * string(maOrder))
			println("True ranking = " * string(trueRanking))
			pValVec = Array(Float64, numIter)
			dmRankVec = Array(Int, numIter)
			for n = 1:numIter
				(lD, ranking) = simlD(maOrder, [elD])
				lD = vec(lD)
				(pValVec[n], dmRankVec[n]) = dm(lD, method=dmM)
			end
			println("    Fail to reject proportion = " * string(sum(dmRankVec .== 0) / numIter))
			println("    Reject in favour of model 1 = " * string(sum(dmRankVec .== 1) / numIter))
			println("    Reject in favour of model 2 = " * string(sum(dmRankVec .== -1) / numIter))
# 			println("    p-value list:")
# 			for n = 1:numIter
# 				println("        " * string(pValVec[n]))
# 			end
		end
	end
	return(true)
end












function simmodel(maOrder::Int, eVarMin::Float64=0.1, eVarMax::Float64=0.25, baseCaseMult::Float64=1.0)
	maOrder == 0 ? maCoef = Array(Float64, 0) : maCoef = [ 1 / 2^p for p = 1:maOrder ]
	xTrue = simulate(numObs, ARIMAModel(maCoef=maCoef))
	vbc = eVarMin + (eVarMax - eVarMin) * rand()
	vbc = baseCaseMult * vbc
	vModel = eVarMin + (eVarMax - eVarMin) * rand(numModel)
	bc = xTrue + vbc * randn(numObs)
	x = [ xTrue + vModel[n] * randn(numObs) for n = 1:numModel ]
	ranking = sortperm([vModel, vbc])
	ranking[ranking .== numModel+1] = 0
	return(xTrue, vbc, bc, vModel, x, ranking)
end
col2mat{T}(x::Vector{Vector{T}}) = [ x[m][n] for n = 1:length(x[1]), m = 1:length(x) ]
mat2col{T}(x::Matrix{T}) = [ x[:, m] for m = 1:size(x, 2) ]










































